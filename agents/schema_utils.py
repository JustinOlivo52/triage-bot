"""
agents/schema_utils.py — Defensive validation for structured LLM output.

Current-generation models can occasionally emit a list field as a JSON-encoded
string (`'["a", "b"]'`) instead of a native array inside otherwise well-formed
structured output. Strict Pydantic validation then rejects a response that is
correct in substance over a pure encoding quirk.

Caught running the seed-cohort build against a live key for the first time:
`recommended_interventions` came back as a string containing a JSON array
rather than the array itself, and the triage call failed validation on an
otherwise good response.
"""

import json
from typing import Any


def coerce_json_encoded_list(value: Any) -> Any:
    """
    If `value` is a string, try to JSON-decode it. Otherwise pass through
    unchanged.

    Used as a `mode="before"` field validator. Deliberately narrow: a string
    that isn't valid JSON is passed through as-is, so normal validation still
    reports whatever is actually wrong rather than this silently swallowing a
    real error.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
