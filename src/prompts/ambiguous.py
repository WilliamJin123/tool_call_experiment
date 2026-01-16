"""Ambiguous evaluation prompts - requiring clarification or interpretation."""

from src.prompts.base import EvalPrompt, ExpectedToolCall

AMBIGUOUS_PROMPTS = [
    EvalPrompt(
        id="ambig_files",
        prompt="Help me with the files",
        expected_calls=[],  # Should ask for clarification
        category="ambiguous",
        description="Vague request about files - needs clarification",
        requires_clarification=True,
    ),
    EvalPrompt(
        id="ambig_process",
        prompt="Process this data",
        expected_calls=[],  # Should ask for clarification
        category="ambiguous",
        description="Vague data processing request - no data provided",
        requires_clarification=True,
    ),
    EvalPrompt(
        id="ambig_send",
        prompt="Send a message",
        expected_calls=[],  # Should ask for clarification
        category="ambiguous",
        description="Missing recipient and content",
        requires_clarification=True,
    ),
    EvalPrompt(
        id="ambig_weather",
        prompt="What's the weather like?",
        expected_calls=[],  # Should ask for location
        category="ambiguous",
        description="Weather query without location",
        requires_clarification=True,
    ),
    EvalPrompt(
        id="ambig_translate",
        prompt="Translate this",
        expected_calls=[],  # Should ask for text and target language
        category="ambiguous",
        description="Translation request without text or target",
        requires_clarification=True,
    ),
    EvalPrompt(
        id="ambig_calculate",
        prompt="Do some math",
        expected_calls=[],  # Should ask for expression
        category="ambiguous",
        description="Math request without expression",
        requires_clarification=True,
    ),
    EvalPrompt(
        id="ambig_partial_email",
        prompt="Send an email to john@example.com",
        expected_calls=[
            ExpectedToolCall(
                name="send_email",
                arguments={"to": "john@example.com"},
                partial_match=True,
            )
        ],
        category="ambiguous",
        description="Email with recipient but missing subject/body - may proceed or ask",
        requires_clarification=True,  # Model should ideally ask for subject/body
    ),
    EvalPrompt(
        id="ambig_search",
        prompt="Find something",
        expected_calls=[],
        category="ambiguous",
        description="Vague search request",
        requires_clarification=True,
    ),
]
