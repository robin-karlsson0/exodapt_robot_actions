import os
import sys

import pytest
from reply_action.reply_action_2 import LANG_CODE_KEY  # noqa: E402
from reply_action.reply_action_2 import (RESULT_KEY, JSONStreamParser,
                                         StreamingState)

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
#                                 'reply_action'))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed(parser: JSONStreamParser, text: str, chunk_size: int = 1) -> str:
    """Feed *text* to *parser* in chunks of *chunk_size*, return all output."""
    result = ""
    for i in range(0, len(text), chunk_size):
        result += parser.process_chunk(text[i:i + chunk_size])
    return result


def _feed_tokens(parser: JSONStreamParser, tokens: list[str]) -> str:
    """Feed an explicit list of token strings and return accumulated output."""
    return "".join(parser.process_chunk(t) for t in tokens)


# ---------------------------------------------------------------------------
# Two-field format: {"lang_code": "...", "reply": "..."}
# ---------------------------------------------------------------------------


class TestJSONStreamParserTwoFieldFormat:
    """Tests for the multilingual two-field JSON format."""

    def setup_method(self):
        self.parser = JSONStreamParser()

    # --- Content extraction ---

    def test_full_content_single_chunk(self):
        content = self.parser.process_chunk(
            '{"lang_code": "en", "reply": "Hello world!"}')
        assert content == "Hello world!"

    def test_japanese_reply_content(self):
        content = self.parser.process_chunk(
            '{"lang_code": "jp", "reply": "こんにちは！本日はどのようなご用件でしょうか？"}')
        assert content == "こんにちは！本日はどのようなご用件でしょうか？"

    def test_unicode_reply_content(self):
        content = self.parser.process_chunk(
            '{"lang_code": "zh", "reply": "你好！今天有什么需要帮助的吗？"}')
        assert content == "你好！今天有什么需要帮助的吗？"

    # --- lang_code extraction ---

    def test_lang_code_extracted_en(self):
        self.parser.process_chunk('{"lang_code": "en", "reply": "Hello!"}')
        assert self.parser.detected_lang_code == "en"

    def test_lang_code_extracted_jp(self):
        self.parser.process_chunk('{"lang_code": "jp", "reply": "こんにちは！"}')
        assert self.parser.detected_lang_code == "jp"

    def test_lang_code_extracted_zh(self):
        self.parser.process_chunk('{"lang_code": "zh", "reply": "你好！"}')
        assert self.parser.detected_lang_code == "zh"

    # --- Streaming simulation ---

    def test_char_by_char_content(self):
        full = '{"lang_code": "en", "reply": "Hello world!"}'
        content = _feed(self.parser, full, chunk_size=1)
        assert content == "Hello world!"

    def test_char_by_char_lang_code(self):
        full = '{"lang_code": "en", "reply": "Hello world!"}'
        _feed(self.parser, full, chunk_size=1)
        assert self.parser.detected_lang_code == "en"

    def test_realistic_token_streaming(self):
        """Simulate how an LLM streams tokens."""
        tokens = ['{"lang_code":', ' "en",', ' "reply":', ' "Hello world!"}']
        content = _feed_tokens(self.parser, tokens)
        assert content == "Hello world!"
        assert self.parser.detected_lang_code == "en"

    # --- Chunk boundary splits ---

    def test_split_inside_lang_code_key(self):
        content = self.parser.process_chunk('{"lang_')
        content += self.parser.process_chunk('code": "en", "reply": "Hi!"}')
        assert content == "Hi!"
        assert self.parser.detected_lang_code == "en"

    def test_split_inside_lang_code_value(self):
        self.parser.process_chunk('{"lang_code": "j')
        self.parser.process_chunk('p", "reply": "こんにちは"}')
        assert self.parser.detected_lang_code == "jp"

    def test_split_inside_reply_key(self):
        content = self.parser.process_chunk('{"lang_code": "en", "rep')
        content += self.parser.process_chunk('ly": "Hello!"}')
        assert content == "Hello!"

    def test_split_inside_reply_content(self):
        content = self.parser.process_chunk(
            '{"lang_code": "en", "reply": "Hello')
        content += self.parser.process_chunk(' world!"}')
        assert content == "Hello world!"

    # --- State machine ---

    def test_state_json_complete_after_full_input(self):
        self.parser.process_chunk('{"lang_code": "en", "reply": "Done."}')
        assert self.parser.state == StreamingState.JSON_COMPLETE

    def test_trailing_garbage_ignored_after_json_complete(self):
        content = self.parser.process_chunk(
            '{"lang_code": "en", "reply": "Done."} extra garbage')
        assert content == "Done."

    def test_lang_code_fully_detected_before_first_content(self):
        """lang_code must be populated before any reply content is emitted."""
        # The reply key straddles a chunk boundary; first content chunk arrives
        # only in the second token. By that point lang_code is fully captured.
        tokens = ['{"lang_code": "fr", "rep', 'ly": "Bonjour!"}']
        first_content_lang_code = None
        for token in tokens:
            content = self.parser.process_chunk(token)
            if content and first_content_lang_code is None:
                first_content_lang_code = self.parser.detected_lang_code
        assert first_content_lang_code == "fr"

    # --- Escape sequences ---

    def test_escaped_double_quote_in_reply(self):
        # JSON value: Say \"hi\"!  (backslash + double-quote preserved)
        content = self.parser.process_chunk(
            '{"lang_code": "en", "reply": "Say \\"hi\\"!"}')
        assert content == 'Say \\"hi\\"!'

    def test_escaped_backslash_in_reply(self):
        # JSON value: Back\\slash  (two consecutive backslashes)
        content = self.parser.process_chunk(
            r'{"lang_code": "en", "reply": "Back\\slash"}')
        # Parser emits both backslashes verbatim
        assert content == r"Back\\slash"


# ---------------------------------------------------------------------------
# Legacy format: {"reply": "..."}   (no lang_code field)
# ---------------------------------------------------------------------------


class TestJSONStreamParserLegacyFormat:
    """Tests for the single-field legacy JSON format (backward compat)."""

    def setup_method(self):
        self.parser = JSONStreamParser()

    def test_full_content_single_chunk(self):
        content = self.parser.process_chunk('{"reply": "Hello world!"}')
        assert content == "Hello world!"

    def test_no_lang_code_detected(self):
        self.parser.process_chunk('{"reply": "Hello world!"}')
        assert self.parser.detected_lang_code == ""

    def test_char_by_char(self):
        content = _feed(self.parser, '{"reply": "Hello!"}', chunk_size=1)
        assert content == "Hello!"

    def test_realistic_token_streaming(self):
        tokens = ['{"reply":', ' "Hello world!"}']
        content = _feed_tokens(self.parser, tokens)
        assert content == "Hello world!"

    def test_state_json_complete_after_full_input(self):
        self.parser.process_chunk('{"reply": "Done."}')
        assert self.parser.state == StreamingState.JSON_COMPLETE


# ---------------------------------------------------------------------------
# FALLBACK_RAW mode  (non-JSON input)
# ---------------------------------------------------------------------------


class TestJSONStreamParserFallbackRaw:
    """Tests for plain-text (non-JSON) input passing through unchanged."""

    def setup_method(self):
        self.parser = JSONStreamParser()

    def test_raw_text_passes_through(self):
        content = self.parser.process_chunk("Hello world!")
        assert content == "Hello world!"

    def test_state_is_fallback_raw(self):
        self.parser.process_chunk("Hello!")
        assert self.parser.state == StreamingState.FALLBACK_RAW

    def test_no_lang_code_in_fallback(self):
        self.parser.process_chunk("Some raw text here.")
        assert self.parser.detected_lang_code == ""

    def test_subsequent_chunks_all_pass_through(self):
        content = _feed_tokens(self.parser, ["Hello", " ", "world", "!"])
        assert content == "Hello world!"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestJSONStreamParserEdgeCases:
    """Tests for boundary conditions and reset behaviour."""

    def setup_method(self):
        self.parser = JSONStreamParser()

    def test_empty_chunk_returns_empty_string(self):
        assert self.parser.process_chunk("") == ""

    def test_empty_chunk_does_not_change_state(self):
        initial_state = self.parser.state
        self.parser.process_chunk("")
        assert self.parser.state == initial_state

    def test_leading_whitespace_before_json_ignored(self):
        """Whitespace before the opening brace must not trigger FALLBACK_RAW."""
        content = self.parser.process_chunk(
            '  \n{"lang_code": "en", "reply": "Hi!"}')
        assert content == "Hi!"
        assert self.parser.detected_lang_code == "en"

    def test_empty_reply_value_two_field(self):
        content = self.parser.process_chunk('{"lang_code": "en", "reply": ""}')
        assert content == ""
        assert self.parser.detected_lang_code == "en"

    def test_empty_reply_value_legacy(self):
        content = self.parser.process_chunk('{"reply": ""}')
        assert content == ""

    # --- reset() ---

    def test_reset_clears_detected_lang_code(self):
        self.parser.process_chunk('{"lang_code": "en", "reply": "Hi!"}')
        assert self.parser.detected_lang_code == "en"
        self.parser.reset()
        assert self.parser.detected_lang_code == ""

    def test_reset_restores_waiting_state(self):
        self.parser.process_chunk('{"lang_code": "en", "reply": "Hi!"}')
        self.parser.reset()
        assert self.parser.state == StreamingState.WAITING_FOR_JSON_START

    def test_reuse_after_reset_two_field(self):
        self.parser.process_chunk('{"lang_code": "en", "reply": "First."}')
        self.parser.reset()
        content = self.parser.process_chunk(
            '{"lang_code": "jp", "reply": "Second."}')
        assert content == "Second."
        assert self.parser.detected_lang_code == "jp"

    def test_reuse_after_reset_legacy(self):
        self.parser.process_chunk('{"reply": "First."}')
        self.parser.reset()
        content = self.parser.process_chunk('{"reply": "Second."}')
        assert content == "Second."
        assert self.parser.detected_lang_code == ""

    def test_lang_code_not_accumulated_across_char_chunks(self):
        """Streaming lang_code char-by-char must not duplicate characters."""
        _feed(self.parser,
              '{"lang_code": "en", "reply": "Test"}',
              chunk_size=1)
        assert self.parser.detected_lang_code == "en"
        assert self.parser.detected_lang_code == "en"
