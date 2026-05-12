from __future__ import annotations

from typing import Any

from recalld.tracing import get_langfuse_client

EVALUATOR_RULE_NAMES_BY_DATASET_KIND = {
    "summary": ("summary-evaluator",),
    "style": ("style-evaluator",),
    "focus": ("focus-evaluator",),
}

LEGACY_EVALUATOR_RULE_NAMES = (
    "theme-guidance-evaluator",
)

DEFAULT_EVALUATOR_RULE_NAMES = (
    "summary-evaluator",
    "style-evaluator",
    "focus-evaluator",
)


def _evaluator_request(client: Any, method: str, path: str, body: Any = None) -> Any:
    httpx_client = client.api._client_wrapper.httpx_client
    response = httpx_client.request(path=path, method=method, json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"Langfuse request failed with HTTP {response.status_code}: {response.text}")
    return response.json()


def _dataset_filter_values(rule: dict[str, Any]) -> list[str]:
    for item in rule.get("filter", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("column") != "datasetId":
            continue
        values = item.get("value")
        if isinstance(values, list):
            return [str(value) for value in values if str(value)]
    return []


def _merge_dataset_filter(rule: dict[str, Any], dataset_id: str) -> tuple[list[dict[str, Any]], bool]:
    filters = [item for item in rule.get("filter", []) or [] if isinstance(item, dict)]
    for item in filters:
        if item.get("column") != "datasetId":
            continue
        values = item.get("value")
        if not isinstance(values, list):
            values = []
            item["value"] = values
        if dataset_id in {str(value) for value in values}:
            return filters, False
        values.append(dataset_id)
        return filters, True

    filters.append(
        {
            "column": "datasetId",
            "type": "stringOptions",
            "operator": "any of",
            "value": [dataset_id],
        }
    )
    return filters, True


def _remove_dataset_filter(rule: dict[str, Any], dataset_id: str) -> tuple[list[dict[str, Any]], bool]:
    filters: list[dict[str, Any]] = []
    changed = False
    for item in rule.get("filter", []) or []:
        if not isinstance(item, dict) or item.get("column") != "datasetId":
            filters.append(item)
            continue
        values = item.get("value")
        if not isinstance(values, list):
            continue
        current_values = [str(value) for value in values if str(value)]
        new_values = [value for value in current_values if value != dataset_id]
        if new_values != current_values:
            changed = True
        if new_values:
            updated = dict(item)
            updated["value"] = new_values
            filters.append(updated)
    return filters, changed


def ensure_evaluator_rules_include_dataset(
    dataset_id: str,
    *,
    dataset_kind: str | None = None,
    rule_names: list[str] | tuple[str, ...] | None = None,
    target: str = "experiment",
    client: Any | None = None,
) -> list[str]:
    client = client or get_langfuse_client()
    if client is None:
        return []

    target_rule_names = tuple(rule_names or EVALUATOR_RULE_NAMES_BY_DATASET_KIND.get(dataset_kind or "", ()))
    if not target_rule_names:
        target_rule_names = DEFAULT_EVALUATOR_RULE_NAMES

    rules = _evaluator_request(client, "GET", "/api/public/unstable/evaluation-rules")
    updated_rule_ids: list[str] = []
    for rule in rules.get("data", []) or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("target") != target:
            continue
        rule_name = rule.get("name")
        if not isinstance(rule_name, str) or not rule_name:
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            continue

        if rule_name in target_rule_names:
            new_filter, changed = _merge_dataset_filter(rule, dataset_id)
        elif rule_name in LEGACY_EVALUATOR_RULE_NAMES:
            if dataset_id not in _dataset_filter_values(rule):
                continue
            new_filter, changed = _remove_dataset_filter(rule, dataset_id)
        else:
            continue

        if not changed:
            continue

        body = {"target": target, "filter": new_filter}
        _evaluator_request(client, "PATCH", f"/api/public/unstable/evaluation-rules/{rule_id}", body)
        updated_rule_ids.append(rule_id)

    return updated_rule_ids
