from __future__ import annotations

from types import SimpleNamespace

from recalld.experiments.langfuse_evaluation_rules import ensure_evaluator_rules_include_dataset


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = "ok"

    def json(self):
        return self._payload


class _FakeHTTPClient:
    def __init__(self, rules: list[dict]):
        self.rules = rules
        self.requests: list[tuple[str, str, dict | None]] = []

    def request(self, *, path: str, method: str, json: dict | None = None):
        self.requests.append((method, path, json))
        if method == "GET":
            return _FakeResponse(200, {"data": self.rules, "meta": {}})
        if method == "PATCH":
            rule_id = path.rsplit("/", 1)[-1]
            for rule in self.rules:
                if rule["id"] == rule_id:
                    rule["filter"] = json["filter"] if json else []
                    rule["target"] = json["target"] if json else rule["target"]
                    return _FakeResponse(200, rule)
            return _FakeResponse(404, {"error": "not found"})
        return _FakeResponse(405, {"error": "unsupported"})


class _FakeClient:
    def __init__(self, rules: list[dict]):
        self.api = SimpleNamespace(_client_wrapper=SimpleNamespace(httpx_client=_FakeHTTPClient(rules)))


def test_ensure_evaluator_rules_include_dataset_updates_matching_rules():
    rules = [
        {
            "id": "rule-1",
            "name": "summary-evaluator",
            "target": "experiment",
            "filter": [
                {
                    "column": "datasetId",
                    "type": "stringOptions",
                    "operator": "any of",
                    "value": ["dataset-old"],
                }
            ],
        },
        {
            "id": "rule-2",
            "name": "style-evaluator",
            "target": "experiment",
            "filter": [],
        },
        {
            "id": "rule-3",
            "name": "focus-evaluator",
            "target": "experiment",
            "filter": [],
        },
        {
            "id": "rule-4",
            "name": "theme-guidance-evaluator",
            "target": "experiment",
            "filter": [
                {
                    "column": "datasetId",
                    "type": "stringOptions",
                    "operator": "any of",
                    "value": ["dataset-new"],
                }
            ],
        },
    ]
    client = _FakeClient(rules)

    updated = ensure_evaluator_rules_include_dataset("dataset-new", dataset_kind="summary", client=client)

    assert updated == ["rule-1", "rule-4"]
    assert rules[0]["filter"][0]["value"] == ["dataset-old", "dataset-new"]
    assert rules[1]["filter"] == []
    assert rules[2]["filter"] == []
    assert rules[3]["filter"] == []


def test_ensure_evaluator_rules_include_dataset_is_idempotent():
    rules = [
        {
            "id": "rule-1",
            "name": "summary-evaluator",
            "target": "experiment",
            "filter": [
                {
                    "column": "datasetId",
                    "type": "stringOptions",
                    "operator": "any of",
                    "value": ["dataset-new"],
                }
            ],
        }
    ]
    client = _FakeClient(rules)

    updated = ensure_evaluator_rules_include_dataset("dataset-new", dataset_kind="summary", client=client)

    assert updated == []
    assert rules[0]["filter"][0]["value"] == ["dataset-new"]


def test_ensure_evaluator_rules_include_dataset_targets_style_only():
    rules = [
        {
            "id": "rule-1",
            "name": "summary-evaluator",
            "target": "experiment",
            "filter": [],
        },
        {
            "id": "rule-2",
            "name": "style-evaluator",
            "target": "experiment",
            "filter": [],
        },
        {
            "id": "rule-3",
            "name": "theme-guidance-evaluator",
            "target": "experiment",
            "filter": [],
        },
    ]
    client = _FakeClient(rules)

    updated = ensure_evaluator_rules_include_dataset("dataset-style", dataset_kind="style", client=client)

    assert updated == ["rule-2"]
    assert rules[0]["filter"] == []
    assert rules[1]["filter"][0]["value"] == ["dataset-style"]
    assert rules[2]["filter"] == []


def test_ensure_evaluator_rules_include_dataset_targets_focus_only():
    rules = [
        {
            "id": "rule-1",
            "name": "summary-evaluator",
            "target": "experiment",
            "filter": [],
        },
        {
            "id": "rule-2",
            "name": "style-evaluator",
            "target": "experiment",
            "filter": [],
        },
        {
            "id": "rule-3",
            "name": "focus-evaluator",
            "target": "experiment",
            "filter": [],
        },
    ]
    client = _FakeClient(rules)

    updated = ensure_evaluator_rules_include_dataset("dataset-focus", dataset_kind="focus", client=client)

    assert updated == ["rule-3"]
    assert rules[0]["filter"] == []
    assert rules[1]["filter"] == []
    assert rules[2]["filter"][0]["value"] == ["dataset-focus"]
