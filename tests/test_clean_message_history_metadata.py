"""Test that _clean_message_history preserves metadata during merge."""
import pytest
from pydantic_ai._agent_graph import _clean_message_history  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.messages import ModelRequest, UserPromptPart, TextPart


def test_clean_message_history_preserves_metadata_on_merge():
    """Test that metadata is preserved when merging consecutive ModelRequest messages."""
    msg1 = ModelRequest(
        parts=[UserPromptPart("Hello")],
        metadata={"session": "abc123", "user_id": "user-456"},
        run_id="run-001",
        conversation_id="conv-789",
    )
    msg2 = ModelRequest(
        parts=[TextPart("Follow-up question")],
        metadata={"session": "abc123", "user_id": "user-456", "extra": "data"},
        run_id="run-001",
        conversation_id="conv-789",
    )

    messages = [msg1, msg2]
    cleaned = _clean_message_history(messages)

    assert len(cleaned) == 1
    merged = cleaned[0]
    
    # Metadata from the last message should be preserved
    assert merged.metadata == {"session": "abc123", "user_id": "user-456", "extra": "data"}
    assert merged.run_id == "run-001"
    assert merged.conversation_id == "conv-789"
    
    # Parts should be merged
    assert len(merged.parts) == 2
    assert isinstance(merged.parts[0], UserPromptPart)
    assert isinstance(merged.parts[1], TextPart)


def test_clean_message_history_falls_back_to_first_metadata():
    """Test metadata fallback when last message has None metadata."""
    msg1 = ModelRequest(
        parts=[UserPromptPart("Hello")],
        metadata={"session": "abc123"},
        run_id="run-001",
    )
    msg2 = ModelRequest(
        parts=[TextPart("Follow-up")],
        metadata=None,
        run_id="run-001",
    )

    messages = [msg1, msg2]
    cleaned = _clean_message_history(messages)

    assert len(cleaned) == 1
    merged = cleaned[0]
    
    # Should fall back to first message's metadata
    assert merged.metadata == {"session": "abc123"}
    assert merged.run_id == "run-001"
