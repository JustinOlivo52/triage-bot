"""
Tests for agents/schema_utils.py.

Regression coverage for a real failure: running the seed-cohort build against
a live key for the first time, a structured triage response came back with
`recommended_interventions` as the string `'["Immediate room placement", ...]'`
instead of an actual list, and strict Pydantic validation rejected an
otherwise-correct response.
"""

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from agents.schema_utils import coerce_json_encoded_list


class TestCoerceJsonEncodedList:

    def test_json_encoded_string_is_parsed(self):
        assert coerce_json_encoded_list('["a", "b"]') == ["a", "b"]

    def test_actual_list_passes_through_unchanged(self):
        value = ["a", "b"]
        assert coerce_json_encoded_list(value) is value

    def test_non_json_string_passes_through_unchanged(self):
        """Not our job to fix a genuinely malformed response — just this quirk."""
        assert coerce_json_encoded_list("not json at all") == "not json at all"

    def test_json_encoded_object_is_still_parsed(self):
        """We coerce any valid JSON, not just arrays — the field validator
        catches a wrong resulting type."""
        assert coerce_json_encoded_list('{"a": 1}') == {"a": 1}

    def test_non_string_non_list_passes_through(self):
        assert coerce_json_encoded_list(None) is None
        assert coerce_json_encoded_list(42) == 42

    def test_empty_string_passes_through(self):
        """Empty string isn't valid JSON; must not raise."""
        assert coerce_json_encoded_list("") == ""


class TestFieldValidatorIntegration:
    """
    The exact failure mode: a schema with a `list[str]` field validated
    against the coercion function via `mode="before"`.
    """

    class _Schema(BaseModel):
        items: list[str]
        _coerce = field_validator("items", mode="before")(coerce_json_encoded_list)

    def test_json_encoded_string_field_validates(self):
        obj = self._Schema.model_validate({"items": '["x", "y"]'})
        assert obj.items == ["x", "y"]

    def test_native_list_field_still_validates(self):
        obj = self._Schema.model_validate({"items": ["x", "y"]})
        assert obj.items == ["x", "y"]

    def test_genuinely_wrong_type_still_raises(self):
        """The fix must not paper over an actually malformed response."""
        with pytest.raises(ValidationError):
            self._Schema.model_validate({"items": "not a list"})

    def test_exact_regression_payload(self):
        """The literal shape seen in production: a JSON array of sentences."""
        payload = (
            '["Immediate room placement for isolation and monitoring", '
            '"Continuous pulse oximetry", "IV access and fluid resuscitation"]'
        )
        obj = self._Schema.model_validate({"items": payload})
        assert len(obj.items) == 3
        assert obj.items[0].startswith("Immediate room placement")
