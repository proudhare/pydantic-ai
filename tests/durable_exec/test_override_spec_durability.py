"""Test that override(spec=...) with capabilities correctly rejects inside durable context."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec.prefect import PrefectDurability
from pydantic_ai.exceptions import UserError


def test_override_spec_with_capabilities_rejects_in_durable_context():
    """Test that override(spec=...) with capabilities raises UserError inside durable context."""
    # Create an agent with PrefectDurability
    durability = PrefectDurability()
    agent = Agent('test', capability=durability)
    
    # Mock the in_durable_context to return True
    with patch.object(durability, 'in_durable_context', return_value=True):
        # Try to override with a spec that has capabilities
        spec = AgentSpec(capabilities=[])
        
        with pytest.raises(UserError) as exc_info:
            with agent.override(spec=spec):
                pass
        
        # Check the error message mentions the key points
        error_msg = str(exc_info.value)
        assert 'override(spec=...)' in error_msg
        assert 'durable' in error_msg.lower()
        assert 'capability' in error_msg.lower()


def test_override_spec_with_capabilities_allowed_outside_durable_context():
    """Test that override(spec=...) with capabilities works outside durable context."""
    # Create an agent with PrefectDurability
    durability = PrefectDurability()
    agent = Agent('test', capability=durability)
    
    # Mock the in_durable_context to return False
    with patch.object(durability, 'in_durable_context', return_value=False):
        # This should work fine
        spec = AgentSpec(capabilities=[])
        
        with agent.override(spec=spec):
            pass  # Should not raise


def test_override_spec_without_capabilities_allowed_in_durable_context():
    """Test that override(spec=...) without capabilities works in durable context."""
    # Create an agent with PrefectDurability
    durability = PrefectDurability()
    agent = Agent('test', capability=durability)
    
    # Mock the in_durable_context to return True
    with patch.object(durability, 'in_durable_context', return_value=True):
        # Override with a spec that has no capabilities should work
        spec = AgentSpec(instructions='new instructions')
        
        with agent.override(spec=spec):
            pass  # Should not raise


def test_override_without_spec_allowed_in_durable_context():
    """Test that override with individual parameters works in durable context."""
    # Create an agent with PrefectDurability
    durability = PrefectDurability()
    agent = Agent('test', capability=durability)
    
    # Mock the in_durable_context to return True
    with patch.object(durability, 'in_durable_context', return_value=True):
        # Override with individual parameters should work fine
        with agent.override(instructions='new instructions'):
            pass  # Should not raise
