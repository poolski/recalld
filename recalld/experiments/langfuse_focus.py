from __future__ import annotations

import argparse
import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langfuse import Evaluation

from recalld.config import load_config
from recalld.experiments import experiment_description
from recalld.experiments import prompt_version as get_prompt_version
from recalld.experiments.langfuse_evaluation_rules import ensure_evaluator_rules_include_dataset
from recalld.experiments.langfuse_session_scores import mirror_experiment_scores_to_session
from recalld.experiments.langfuse_summary import load_summary_experiment_context
from recalld.jobs import DEFAULT_SCRATCH_ROOT
from recalld.llm.context import ensure_loaded_context_length, list_available_models, token_budget
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.focus import generate_focus_points
from recalld.tracing import (
    experiment_tracing_environment,
    get_langfuse_client,
    job_session_token,
    make_session_id,
    session_context,
    shutdown_tracing,
    update_current_span,
)

DEFAULT_DATASET_SUFFIX = "focus"
DEFAULT_PROMPT_LABELS = ("production",)
FOCUS_PROMPT_NAME = "recalld/focus-instructions"


@dataclass(frozen=True)
class FocusExperimentContext:
    job_id: str
    dataset_name: str
    dataset_item_id: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    original_filename: str = ""


def load_focus_experiment_context(
    job_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> FocusExperimentContext:
    summary_ctx = load_summary_experiment_context(job_id, scratch_root=scratch_root)
    dataset_name = f"recalld/jobs/{job_id}/{DEFAULT_DATASET_SUFFIX}"
    dataset_item_id = f"job-{job_id}-focus"
    return FocusExperimentContext(
        job_id=summary_ctx.job_id,
        dataset_name=dataset_name,
        dataset_item_id=dataset_item_id,
        input_data=summary_ctx.input_data,
        expected_output={"focus_points": summary_ctx.expected_output.get("focus_points", [])},
        original_filename=summary_ctx.original_filename,
    )


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    return " ".join(text.split())


def _focus_overlap(candidate: Any, reference: Any) -> float:
    candidate_items = {_normalize_text(item) for item in candidate or [] if _normalize_text(item)}
    reference_items = {_normalize_text(item) for item in reference or [] if _normalize_text(item)}
    if not reference_items:
        return 1.0 if not candidate_items else 0.0
    if not candidate_items:
        return 0.0
    return len(candidate_items & reference_items) / len(reference_items)


def _count_similarity(candidate: Any, reference: Any) -> float:
    candidate_items = list(candidate or [])
    reference_items = list(reference or [])
    if not reference_items:
        return 1.0 if not candidate_items else 0.0
    if not candidate_items:
        return 0.0
    largest = max(len(candidate_items), len(reference_items))
    return max(0.0, 1.0 - (abs(len(candidate_items) - len(reference_items)) / largest))


def make_focus_alignment_evaluator(prompt_label: str):
    def evaluator(*, input, output, expected_output, metadata, **kwargs) -> Evaluation:
        candidate_focus = []
        if isinstance(output, dict):
            candidate_focus = output.get("focus_points", []) or []
        reference_focus = []
        if isinstance(expected_output, dict):
            reference_focus = expected_output.get("focus_points", []) or []
        focus_score = _focus_overlap(candidate_focus, reference_focus)
        count_score = _count_similarity(candidate_focus, reference_focus)
        overall = round((focus_score * 0.8) + (count_score * 0.2), 3)
        comment = (
            f"focus_overlap={focus_score:.3f}; "
            f"count_similarity={count_score:.3f}; "
            f"prompt_label={prompt_label}"
        )
        return Evaluation(name="reference_alignment", value=overall, comment=comment)

    evaluator.__name__ = f"focus_alignment_evaluator[{prompt_label}]"
    return evaluator


async def _generate_focus_points(
    *,
    input_data: dict[str, Any],
    prompt_label: str,
    llm_base_url: str,
    llm_model: str,
    token_budget_value: int,
) -> dict[str, Any]:
    turns = [LabelledTurn(**turn) for turn in input_data["aligned_turns"]]
    result = await generate_focus_points(
        turns=turns,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        token_budget=token_budget_value,
        speaker_a_name=input_data.get("speaker_a_name", "You"),
        speaker_b_name=input_data.get("speaker_b_name", "Speaker 2"),
        prompt_label=prompt_label,
    )
    return {
        "focus_points": result.focus_points,
        "strategy": result.strategy,
        "topic_count": result.topic_count,
    }


def _run_name(
    step: str,
    original_filename: str,
    prompt_label: str,
    llm_model: str,
    context_length: int,
    run_tag: str | None,
) -> str:
    stem = Path(original_filename).stem if original_filename else "job"
    parts = [step, stem, prompt_label, llm_model, f"ctx{context_length}"]
    if run_tag:
        parts.append(run_tag)
    parts.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    return " · ".join(parts)


def _load_runtime_config() -> tuple[str, str, float]:
    cfg = load_config()
    return cfg.llm_base_url, cfg.llm_model, cfg.llm_context_headroom


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


def _ensure_dataset(client, context: FocusExperimentContext):
    try:
        dataset = client.get_dataset(context.dataset_name)
        existing_ids = {item.id for item in dataset.items}
    except Exception:
        client.create_dataset(
            name=context.dataset_name,
            description=f"Focus sub-prompt experiment for job {context.job_id}",
            metadata={
                "job_id": context.job_id,
                "type": "focus_prompt_experiment",
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
                "type": "focus_prompt_experiment",
            },
        )
    dataset = client.get_dataset(context.dataset_name)
    ensure_evaluator_rules_include_dataset(dataset.id, dataset_kind="focus", client=client)
    return dataset


def run_focus_prompt_experiment(
    *,
    job_id: str,
    prompt_labels: list[str] | None = None,
    llm_models: list[str] | None = None,
    context_length: int | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    dataset_name: str | None = None,
    clone_from_label: str | None = None,
    run_tag: str | None = None,
    experiment_name: str | None = None,
) -> list[dict[str, Any]]:
    with experiment_tracing_environment():
        try:
            client = get_langfuse_client()
            if client is None:
                raise RuntimeError("Langfuse credentials are unavailable; cannot run experiment.")

            context = load_focus_experiment_context(job_id, scratch_root=scratch_root)
            if dataset_name:
                context = FocusExperimentContext(
                    job_id=context.job_id,
                    dataset_name=dataset_name,
                    dataset_item_id=context.dataset_item_id,
                    input_data=context.input_data,
                    expected_output=context.expected_output,
                    original_filename=context.original_filename,
                )

            dataset = _ensure_dataset(client, context)
            labels = prompt_labels or list(DEFAULT_PROMPT_LABELS)
            llm_base_url, fallback_model, headroom = _load_runtime_config()
            model_ids = llm_models or ([fallback_model] if fallback_model else [])
            if not model_ids:
                raise ValueError("No LM Studio model ids were provided.")
            if llm_models is not None:
                available = asyncio.run(list_available_models(llm_base_url, selected_model=model_ids[0]))
                available_ids = {model.id for model in available}
                missing = [model_id for model_id in model_ids if model_id not in available_ids]
                if missing:
                    raise ValueError(f"Requested LM Studio model(s) not available: {', '.join(missing)}")
            run_results: list[dict[str, Any]] = []

            if clone_from_label:
                from recalld.experiments.langfuse_summary import clone_prompt_label

                for prompt_label in labels:
                    if prompt_label == clone_from_label:
                        continue
                    clone_prompt_label(
                        source_label=clone_from_label,
                        target_label=prompt_label,
                        prompt_names=(FOCUS_PROMPT_NAME,),
                        client=client,
                    )

            for model_id in model_ids:
                target_context_length = asyncio.run(
                    ensure_loaded_context_length(
                        llm_base_url,
                        model_id,
                        requested_context_length=context_length,
                    )
                )
                target_budget = token_budget(target_context_length, headroom)
                for prompt_label in labels:
                    run_name = _run_name("focus", context.original_filename, prompt_label, model_id, target_context_length, run_tag)
                    run_session_id = make_session_id(
                        "focus",
                        job_session_token(context.job_id) or context.job_id,
                        prompt_label,
                        model_id,
                        f"ctx{target_context_length}",
                        run_tag,
                        timestamp=datetime.now(timezone.utc),
                    )

                    def task(*, item, **kwargs):
                        update_current_span(name=f"focus · {filename} · {prompt_label} · {model_id}")
                        return _run_coro_sync(
                            _generate_focus_points(
                                input_data=item.input,
                                prompt_label=prompt_label,
                                llm_base_url=llm_base_url,
                                llm_model=model_id,
                                token_budget_value=target_budget,
                            )
                        )

                    tag_parts = [f"[{run_tag}]"] if run_tag else []
                    default_name = " ".join([model_id, prompt_label] + tag_parts)
                    exp_name = experiment_name or default_name
                    filename = Path(context.original_filename).stem if context.original_filename else context.job_id
                    pv = get_prompt_version(client, FOCUS_PROMPT_NAME, prompt_label)
                    exp_description = experiment_description(
                        exp_type="focus",
                        filename=filename,
                        prompt_label=prompt_label,
                        prompt_version=pv,
                        llm_model=model_id,
                        all_labels=labels,
                        run_tag=run_tag,
                    )
                    meta: dict[str, Any] = {
                        "job_id": context.job_id,
                        "dataset_name": context.dataset_name,
                        "prompt_label": prompt_label,
                        "llm_model": model_id,
                        "context_length": target_context_length,
                        "experiment_name": "focus",
                    }
                    if run_tag:
                        meta["run_tag"] = run_tag
                    with session_context(run_session_id):
                        result = dataset.run_experiment(
                            name=exp_name,
                            run_name=run_name,
                            description=exp_description,
                            task=task,
                            evaluators=[make_focus_alignment_evaluator(prompt_label)],
                            max_concurrency=1,
                            metadata=meta,
                        )
                    mirror_experiment_scores_to_session(
                        client,
                        result,
                        session_id=run_session_id,
                        metadata=meta,
                        wait_for_scores=True,
                        settle_timeout_seconds=10.0,
                    )
                    run_results.append(
                        {
                            "prompt_label": prompt_label,
                            "llm_model": model_id,
                            "context_length": target_context_length,
                            "run_name": run_name,
                            "result": result,
                        }
                    )

            return run_results
        finally:
            shutdown_tracing()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Langfuse focus prompt experiment for a saved recalld job."
    )
    parser.add_argument("--job-id", required=True, help="Saved recalld job id")
    parser.add_argument(
        "--prompt-label",
        action="append",
        dest="prompt_labels",
        help="Langfuse prompt label to compare; repeat for multiple labels",
    )
    parser.add_argument(
        "--llm-model",
        action="append",
        dest="llm_models",
        help="LM Studio model id to run; repeat for multiple models",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=32768,
        help="Context length to request when loading each model",
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
    results = run_focus_prompt_experiment(
        job_id=args.job_id,
        prompt_labels=args.prompt_labels,
        llm_models=args.llm_models,
        context_length=args.context_length,
        scratch_root=Path(args.scratch_root),
        dataset_name=args.dataset_name,
        clone_from_label=args.clone_from_label,
    )
    for item in results:
        print(f"{item['prompt_label']} [{item['llm_model']}]: {item['run_name']}")
        print(item["result"].format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
