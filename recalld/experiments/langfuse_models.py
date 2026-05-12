from __future__ import annotations

from dataclasses import dataclass

from recalld.llm.context import ensure_loaded_context_length, list_available_models, token_budget


@dataclass(frozen=True)
class ModelExperimentTarget:
    llm_model: str
    context_length: int
    token_budget: int


async def resolve_model_targets(
    *,
    base_url: str,
    fallback_model: str,
    requested_models: list[str] | None,
    requested_context_length: int | None,
    headroom: float,
    load_context_length_fn=ensure_loaded_context_length,
    budget_fn=token_budget,
) -> list[ModelExperimentTarget]:
    model_ids = requested_models or ([fallback_model] if fallback_model else [])
    if not model_ids:
        raise ValueError("No LM Studio model ids were provided.")

    if requested_models is not None:
        available = await list_available_models(base_url, selected_model=model_ids[0])
        available_ids = {model.id for model in available}
        missing = [model_id for model_id in model_ids if model_id not in available_ids]
        if missing:
            raise ValueError(f"Requested LM Studio model(s) not available: {', '.join(missing)}")

    targets: list[ModelExperimentTarget] = []
    for model_id in model_ids:
        context_length = await load_context_length_fn(
            base_url,
            model_id,
            requested_context_length=requested_context_length,
        )
        targets.append(
            ModelExperimentTarget(
                llm_model=model_id,
                context_length=context_length,
                token_budget=budget_fn(context_length, headroom),
            )
        )
    return targets
