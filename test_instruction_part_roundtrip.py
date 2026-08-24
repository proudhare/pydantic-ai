# tests/test_instruction_part_roundtrip.py
from __future__ import annotations
import warnings
import pytest
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
)

def test_instruction_part_round_trip_json() -> None:
    request = ModelRequest(parts=[InstructionPart(content='Be helpful.', dynamic=False)])
    serialized = ModelMessagesTypeAdapter.dump_json([request])  # raises PydanticSerializationError
    deserialized = ModelMessagesTypeAdapter.validate_json(serialized)
    assert len(deserialized) == 1
    assert isinstance(deserialized[0].parts[0], InstructionPart)

def test_full_history_round_trip() -> None:
    history = [
        ModelRequest(parts=[InstructionPart(content='Be helpful.', dynamic=False)]),
        ModelResponse(parts=[TextPart(content='OK')]),
    ]
    serialized = ModelMessagesTypeAdapter.dump_json(history)
    deserialized = ModelMessagesTypeAdapter.validate_json(serialized)  # raises ValidationError
    assert deserialized[0].parts[0].content == 'Be helpful.'
