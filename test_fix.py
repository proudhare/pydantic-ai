from pydantic_ai._deferred_capabilities import parse_loaded_capabilities
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelRequest,
    ModelResponse,
)
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

messages = [
    ModelResponse(
        parts=[
            LoadCapabilityCallPart(
                tool_call_id="load-foobar",
                args={"id": "foobar"},
            )
        ]
    ),
    ModelRequest(
        parts=[
            LoadCapabilityReturnPart(
                tool_call_id="load-foobar",
                content={"instructions": "# Foo Bar"},
            )
        ]
    ),
]

assert parse_loaded_capabilities(messages) == {"foobar"}
print("✓ Original messages parse correctly")

ui_messages = VercelAIAdapter.dump_messages(messages)
print(f"✓ Dumped to {len(ui_messages)} UI messages")

round_tripped = VercelAIAdapter.load_messages(ui_messages)
print(f"✓ Loaded {len(round_tripped)} messages back")

# Check the types
for msg in round_tripped:
    for part in msg.parts:
        print(f"  Part type: {type(part).__name__}")

# Expected: {"foobar"}
# Actual: set()
result = parse_loaded_capabilities(round_tripped)
print(f"Loaded capabilities: {result}")
assert result == {"foobar"}, f"Expected {{'foobar'}}, got {result}"
print("✓ Round-tripped messages preserve loaded capability state")
