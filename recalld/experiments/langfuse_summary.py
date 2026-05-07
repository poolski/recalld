from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from langfuse import Evaluation

from recalld.config import load_config
from recalld.experiments import experiment_description
from recalld.experiments import prompt_version as get_prompt_version
from recalld.jobs import DEFAULT_SCRATCH_ROOT, load_job
from recalld.llm.context import ensure_loaded_context_length, token_budget
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import postprocess
from recalld.tracing import get_langfuse_client, shutdown_tracing, update_current_span

DEFAULT_DATASET_SUFFIX = "summary"
DEFAULT_PROMPT_LABELS = ("production",)
SUMMARY_PROMPT_NAMES = (
    "recalld/postprocess-style-analysis",
    "recalld/postprocess-summary-single",
    "recalld/postprocess-summary-reduce",
    "recalld/postprocess-summary-map",
)


@dataclass(frozen=True)
class SummaryExperimentContext:
    job_id: str
    dataset_name: str
    dataset_item_id: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    original_filename: str = ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_aligned_turns(job_id: str, scratch_root: Path) -> list[LabelledTurn]:
    job = load_job(job_id, scratch_root=scratch_root)
    if not job.aligned_path:
        raise ValueError(f"job {job_id} does not have aligned transcript output")
    path = Path(job.aligned_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return [LabelledTurn(**turn) for turn in _read_json(path)]


def load_summary_experiment_context(
    job_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> SummaryExperimentContext:
    job = load_job(job_id, scratch_root=scratch_root)
    aligned_turns = _load_aligned_turns(job_id, scratch_root)

    if not job.postprocess_path:
        raise ValueError(f"job {job_id} does not have postprocess output")
    postprocess_path = Path(job.postprocess_path)
    if not postprocess_path.exists():
        raise FileNotFoundError(postprocess_path)
    reference = _read_json(postprocess_path)
    if not isinstance(reference, dict):
        raise ValueError(f"expected postprocess output for job {job_id} to be a JSON object")

    input_data = {
        "job_id": job.id,
        "category_id": job.category_id,
        "original_filename": job.original_filename,
        "speaker_a_name": job.speaker_00 or "You",
        "speaker_b_name": job.speaker_01 or "Speaker 2",
        "aligned_turns": [turn.__dict__.copy() for turn in aligned_turns],
        "theme_guidance": job.theme_guidance,
        "existing_note_content": "",
    }
    expected_output = {
        "summary": reference.get("summary", ""),
        "focus_points": reference.get("focus_points", []),
        "strategy": reference.get("strategy", ""),
        "topic_count": reference.get("topic_count"),
    }
    dataset_name = f"recalld/jobs/{job.id}/{DEFAULT_DATASET_SUFFIX}"
    dataset_item_id = f"job-{job.id}-summary"
    return SummaryExperimentContext(
        job_id=job.id,
        dataset_name=dataset_name,
        dataset_item_id=dataset_item_id,
        input_data=input_data,
        expected_output=expected_output,
        original_filename=job.original_filename or "",
    )


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _summary_similarity(candidate: str, reference: str) -> float:
    if not reference:
        return 1.0 if not candidate else 0.0
    return SequenceMatcher(None, _normalize_text(candidate), _normalize_text(reference)).ratio()


def _focus_overlap(candidate: Any, reference: Any) -> float:
    candidate_items = {_normalize_text(item) for item in candidate or [] if _normalize_text(item)}
    reference_items = {_normalize_text(item) for item in reference or [] if _normalize_text(item)}
    if not reference_items:
        return 1.0 if not candidate_items else 0.0
    if not candidate_items:
        return 0.0
    return len(candidate_items & reference_items) / len(reference_items)


def make_summary_alignment_evaluator(prompt_label: str):
    def evaluator(*, input, output, expected_output, metadata, **kwargs) -> Evaluation:
        candidate_summary = ""
        candidate_focus = []
        if isinstance(output, dict):
            candidate_summary = str(output.get("summary", ""))
            candidate_focus = output.get("focus_points", []) or []
        else:
            candidate_summary = str(output or "")

        reference_summary = ""
        reference_focus = []
        if isinstance(expected_output, dict):
            reference_summary = str(expected_output.get("summary", ""))
            reference_focus = expected_output.get("focus_points", []) or []
        else:
            reference_summary = str(expected_output or "")

        summary_score = _summary_similarity(candidate_summary, reference_summary)
        focus_score = _focus_overlap(candidate_focus, reference_focus)
        overall = round((summary_score * 0.75) + (focus_score * 0.25), 3)
        comment = (
            f"summary_similarity={summary_score:.3f}; "
            f"focus_overlap={focus_score:.3f}; "
            f"prompt_label={prompt_label}"
        )
        return Evaluation(name="reference_alignment", value=overall, comment=comment)

    evaluator.__name__ = f"summary_alignment_evaluator[{prompt_label}]"
    return evaluator


async def _generate_summary(
    *,
    input_data: dict[str, Any],
    prompt_label: str,
    llm_base_url: str,
    llm_model: str,
    token_budget_value: int,
) -> dict[str, Any]:
    turns = [LabelledTurn(**turn) for turn in input_data["aligned_turns"]]
    result = await postprocess(
        turns=turns,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        token_budget=token_budget_value,
        speaker_a_name=input_data.get("speaker_a_name", "You"),
        speaker_b_name=input_data.get("speaker_b_name", "Speaker 2"),
        existing_note_content=input_data.get("existing_note_content", ""),
        theme_guidance=input_data.get("theme_guidance", []),
        prompt_label=prompt_label,
    )
    return {
        "summary": result.summary,
        "focus_points": result.focus_points,
        "strategy": result.strategy,
        "topic_count": result.topic_count,
    }


def _run_name(original_filename: str, prompt_label: str, run_tag: str | None) -> str:
    stem = Path(original_filename).stem if original_filename else "job"
    parts = [stem, prompt_label]
    if run_tag:
        parts.append(run_tag)
    parts.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    return " · ".join(parts)


def _load_runtime_config() -> tuple[str, str, int]:
    cfg = load_config()
    ctx_len = asyncio.run(ensure_loaded_context_length(cfg.llm_base_url, cfg.llm_model))
    return cfg.llm_base_url, cfg.llm_model, token_budget(ctx_len, cfg.llm_context_headroom)


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def _worker():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive thread bridge
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result.get("value")


def _ensure_dataset(client, context: SummaryExperimentContext):
    try:
        dataset = client.get_dataset(context.dataset_name)
        existing_ids = {item.id for item in dataset.items}
    except Exception:
        client.create_dataset(
            name=context.dataset_name,
            description=f"Prompt tuning experiment for job {context.job_id}",
            metadata={
                "job_id": context.job_id,
                "type": "summary_prompt_experiment",
            },
        )
        existing_ids = set()

    if context.dataset_item_id not in existing_ids:
        client.create_dataset_item(
            dataset_name=context.dataset_name,
            id=context.dataset_item_id,
            input=context.input_data,
            expected_output=context.expected_output,
            metadata={
                "job_id": context.job_id,
                "type": "summary_prompt_experiment",
            },
        )
    return client.get_dataset(context.dataset_name)


def clone_prompt_label(
    *,
    source_label: str,
    target_label: str,
    prompt_names: tuple[str, ...] = SUMMARY_PROMPT_NAMES,
    client=None,
) -> None:
    client = client or get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse credentials are unavailable; cannot clone prompts.")

    for prompt_name in prompt_names:
        prompt = client.get_prompt(prompt_name, label=source_label)
        version = getattr(prompt, "version", None)
        if version is None:
            raise RuntimeError(f"prompt {prompt_name} did not expose a version")
        client.update_prompt(name=prompt_name, version=version, new_labels=[target_label])


def run_summary_prompt_experiment(
    *,
    job_id: str,
    prompt_labels: list[str] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    dataset_name: str | None = None,
    clone_from_label: str | None = None,
    run_tag: str | None = None,
    experiment_name: str | None = None,
) -> list[dict[str, Any]]:
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse credentials are unavailable; cannot run experiment.")

    context = load_summary_experiment_context(job_id, scratch_root=scratch_root)
    if dataset_name:
        context = SummaryExperimentContext(
            job_id=context.job_id,
            dataset_name=dataset_name,
            dataset_item_id=context.dataset_item_id,
            input_data=context.input_data,
            expected_output=context.expected_output,
        )

    dataset = _ensure_dataset(client, context)
    labels = prompt_labels or list(DEFAULT_PROMPT_LABELS)
    llm_base_url, llm_model, budget = _load_runtime_config()
    run_results: list[dict[str, Any]] = []

    if clone_from_label:
        for prompt_label in labels:
            if prompt_label == clone_from_label:
                continue
            clone_prompt_label(
                source_label=clone_from_label,
                target_label=prompt_label,
                client=client,
            )

    for prompt_label in labels:
        run_name = _run_name(context.original_filename, prompt_label, run_tag)

        def task(*, item, **kwargs):
            update_current_span(name=f"summarise · {filename} · {prompt_label}")
            return _run_coro_sync(
                _generate_summary(
                    input_data=item.input,
                    prompt_label=prompt_label,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    token_budget_value=budget,
                )
            )

        tag_parts = [f"[{run_tag}]"] if run_tag else []
        default_name = " ".join([prompt_label] + tag_parts)
        exp_name = experiment_name or default_name
        filename = Path(context.original_filename).stem if context.original_filename else context.job_id
        pv = get_prompt_version(client, "recalld/postprocess-summary-single", prompt_label)
        exp_description = experiment_description(
            exp_type="summary",
            filename=filename,
            prompt_label=prompt_label,
            prompt_version=pv,
            llm_model=llm_model,
            all_labels=labels,
            run_tag=run_tag,
        )
        meta: dict[str, Any] = {
            "job_id": context.job_id,
            "dataset_name": context.dataset_name,
            "prompt_label": prompt_label,
            "llm_model": llm_model,
            "experiment_name": "summary",
        }
        if run_tag:
            meta["run_tag"] = run_tag
        result = dataset.run_experiment(
            name=exp_name,
            run_name=run_name,
            description=exp_description,
            task=task,
            evaluators=[make_summary_alignment_evaluator(prompt_label)],
            max_concurrency=1,
            metadata=meta,
        )
        run_results.append(
            {
                "prompt_label": prompt_label,
                "run_name": run_name,
                "result": result,
            }
        )

    shutdown_tracing()
    return run_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Langfuse summary prompt experiment for a saved recalld job."
    )
    parser.add_argument("--job-id", required=True, help="Saved recalld job id")
    parser.add_argument(
        "--prompt-label",
        action="append",
        dest="prompt_labels",
        help="Langfuse prompt label to compare; repeat for multiple labels",
    )
    parser.add_argument(
        "--dataset-name",
        help="Override the Langfuse dataset name",
    )
    parser.add_argument(
        "--clone-from-label",
        help="Clone prompts from this existing label before running other labels",
    )
    parser.add_argument(
        "--scratch-root",
        default=str(DEFAULT_SCRATCH_ROOT),
        help="Path to the recalld scratch root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results = run_summary_prompt_experiment(
        job_id=args.job_id,
        prompt_labels=args.prompt_labels,
        scratch_root=Path(args.scratch_root),
        dataset_name=args.dataset_name,
        clone_from_label=args.clone_from_label,
    )
    for item in results:
        print(f"{item['prompt_label']}: {item['run_name']}")
        print(item["result"].format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
