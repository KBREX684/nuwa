"""Unit tests for nuwa.llm.response_parser."""

from __future__ import annotations

import pytest

from nuwa.core.exceptions import LLMError
from nuwa.llm.response_parser import parse_json_response


class TestParseJsonResponse:
    """Tests for parse_json_response covering various LLM output quirks."""

    def test_plain_json_object(self) -> None:
        raw = '{"key": "value", "num": 42}'
        result = parse_json_response(raw)
        assert result == {"key": "value", "num": 42}

    def test_plain_json_array(self) -> None:
        raw = "[1, 2, 3]"
        result = parse_json_response(raw)
        assert result == [1, 2, 3]

    def test_json_in_markdown_code_fence(self) -> None:
        raw = 'Here is the result:\n```json\n{"score": 0.9}\n```\nDone.'
        result = parse_json_response(raw)
        assert result == {"score": 0.9}

    def test_json_in_code_fence_without_lang(self) -> None:
        raw = '```\n{"score": 0.8}\n```'
        result = parse_json_response(raw)
        assert result == {"score": 0.8}

    def test_trailing_commas_removed(self) -> None:
        raw = '{"a": 1, "b": 2,}'
        result = parse_json_response(raw)
        assert result == {"a": 1, "b": 2}

    def test_trailing_commas_in_array(self) -> None:
        raw = '{"items": [1, 2, 3,]}'
        result = parse_json_response(raw)
        assert result == {"items": [1, 2, 3]}

    def test_single_line_comments_stripped(self) -> None:
        raw = '{\n  "key": "value", // this is a comment\n  "num": 42\n}'
        result = parse_json_response(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_embedded_in_prose(self) -> None:
        raw = 'The analysis shows:\n{"diagnosis": "good"}\nEnd of analysis.'
        result = parse_json_response(raw)
        assert result == {"diagnosis": "good"}

    def test_array_before_object(self) -> None:
        raw = 'Result: [{"id": 1}, {"id": 2}] and then {}'
        result = parse_json_response(raw)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_nested_braces(self) -> None:
        raw = '{"outer": {"inner": "val"}, "count": 1}'
        result = parse_json_response(raw)
        assert result == {"outer": {"inner": "val"}, "count": 1}

    def test_empty_string_raises(self) -> None:
        with pytest.raises(LLMError, match="empty response"):
            parse_json_response("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(LLMError, match="empty response"):
            parse_json_response("   \n\t  ")

    def test_non_json_raises(self) -> None:
        with pytest.raises(LLMError, match="Failed to parse JSON"):
            parse_json_response("This is just plain text with no JSON at all.")

    def test_boolean_values(self) -> None:
        raw = '{"active": true, "disabled": false}'
        result = parse_json_response(raw)
        assert result == {"active": True, "disabled": False}

    def test_null_values(self) -> None:
        raw = '{"value": null}'
        result = parse_json_response(raw)
        assert result == {"value": None}

    def test_unicode_content(self) -> None:
        raw = '{"message": "你好世界", "score": 0.9}'
        result = parse_json_response(raw)
        assert result["message"] == "你好世界"

    def test_escaped_quotes_in_values(self) -> None:
        raw = '{"text": "He said \\"hello\\""}'
        result = parse_json_response(raw)
        assert result["text"] == 'He said "hello"'

    def test_large_nested_structure(self) -> None:
        raw = '{"mutations": [{"type": "config_change", "value": {"a": 1}}, {"type": "prompt"}]}'
        result = parse_json_response(raw)
        assert len(result["mutations"]) == 2
