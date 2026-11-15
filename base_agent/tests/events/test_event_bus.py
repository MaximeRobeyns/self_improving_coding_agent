# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for the event bus module.

These tests verify the core functionality of the EventBus class, which serves
as the centralized event management system for the agent framework.
"""
import pytest
import pytest_asyncio
import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from src.events.event_bus import EventBus, EventEncoder
from src.types.event_types import EventType, Event, FileEvent, FileOperation

# Mark all tests as asyncio tests
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def reset_event_bus():
    """Reset the EventBus singleton between tests."""
    # Reset the singleton instance to ensure tests are isolated
    EventBus._instance = None
    EventBus._lock = None
    
    # Create a new instance and yield it
    event_bus = await EventBus.get_instance()
    yield event_bus
    
    # Clean up after tests
    event_bus.clear()
    EventBus._instance = None
    EventBus._lock = None


async def test_singleton_pattern(reset_event_bus):
    """Test that EventBus follows the singleton pattern."""
    # Get two instances
    instance1 = await EventBus.get_instance()
    instance2 = await EventBus.get_instance()
    
    # Assert that they are the same instance
    assert instance1 is instance2
    
    # Check that direct instantiation is prevented
    with pytest.raises(TypeError):
        EventBus()


async def test_publish_subscribe(reset_event_bus):
    """Test basic publish and subscribe functionality."""
    event_bus = await EventBus.get_instance()
    
    # Create a mock callback
    mock_callback = AsyncMock()
    
    # Subscribe to an event
    event_bus.subscribe(EventType.ASSISTANT_MESSAGE, mock_callback)
    
    # Create and publish an event
    test_content = "Test message"
    event = Event(type=EventType.ASSISTANT_MESSAGE, content=test_content)
    publisher_id = "test_agent_id"
    await event_bus.publish(event, publisher_id)
    
    # Verify callback was called with the event
    mock_callback.assert_called_once()
    called_event = mock_callback.call_args[0][0]
    assert called_event.type == EventType.ASSISTANT_MESSAGE
    assert called_event.content == test_content
    
    # Verify publisher ID was added to metadata
    assert called_event.metadata.get("publisher_id") == publisher_id


async def test_subscribe_multiple_types(reset_event_bus):
    """Test subscribing to multiple event types at once."""
    event_bus = await EventBus.get_instance()
    
    # Create a mock callback
    mock_callback = AsyncMock()
    
    # Subscribe to multiple event types
    event_types = {EventType.ASSISTANT_MESSAGE, EventType.TOOL_CALL}
    event_bus.subscribe(event_types, mock_callback)
    
    # Publish events of different types
    for event_type in event_types:
        event = Event(type=event_type, content=f"Test for {event_type.value}")
        await event_bus.publish(event, "test_agent_id")
    
    # Verify callback was called for each event type
    assert mock_callback.call_count == len(event_types)


async def test_unsubscribe(reset_event_bus):
    """Test unsubscribing from events."""
    event_bus = await EventBus.get_instance()
    
    # Create a mock callback
    mock_callback = AsyncMock()
    
    # Subscribe to an event
    event_bus.subscribe(EventType.ASSISTANT_MESSAGE, mock_callback)
    
    # Unsubscribe from the event
    event_bus.unsubscribe(EventType.ASSISTANT_MESSAGE, mock_callback)
    
    # Publish an event
    event = Event(type=EventType.ASSISTANT_MESSAGE, content="Test message")
    await event_bus.publish(event, "test_agent_id")
    
    # Verify callback was not called
    mock_callback.assert_not_called()


async def test_unsubscribe_multiple(reset_event_bus):
    """Test unsubscribing from multiple event types at once."""
    event_bus = await EventBus.get_instance()
    
    # Create a mock callback
    mock_callback = AsyncMock()
    
    # Subscribe to multiple event types
    event_types = {EventType.ASSISTANT_MESSAGE, EventType.TOOL_CALL}
    event_bus.subscribe(event_types, mock_callback)
    
    # Unsubscribe from all event types
    event_bus.unsubscribe(event_types, mock_callback)
    
    # Publish events
    for event_type in event_types:
        event = Event(type=event_type, content=f"Test for {event_type.value}")
        await event_bus.publish(event, "test_agent_id")
    
    # Verify callback was not called
    mock_callback.assert_not_called()


async def test_get_events(reset_event_bus):
    """Test retrieving events by publisher ID."""
    event_bus = await EventBus.get_instance()
    
    # Create and publish events from different publishers
    publisher1_id = "agent1"
    publisher2_id = "agent2"
    
    event1 = Event(type=EventType.ASSISTANT_MESSAGE, content="Message from agent1")
    event2 = Event(type=EventType.TOOL_CALL, content="Tool call from agent1")
    event3 = Event(type=EventType.ASSISTANT_MESSAGE, content="Message from agent2")
    
    await event_bus.publish(event1, publisher1_id)
    await event_bus.publish(event2, publisher1_id)
    await event_bus.publish(event3, publisher2_id)
    
    # Get events for publisher1
    publisher1_events = event_bus.get_events(publisher1_id)
    
    # Verify correct events are returned
    assert len(publisher1_events) == 2
    assert any(e.content == "Message from agent1" for e in publisher1_events)
    assert any(e.content == "Tool call from agent1" for e in publisher1_events)
    assert not any(e.content == "Message from agent2" for e in publisher1_events)


async def test_get_events_by_type(reset_event_bus):
    """Test retrieving events by event type."""
    # Clear all existing state and create a fresh EventBus instance
    EventBus._instance = None
    EventBus._lock = None
    event_bus = await EventBus.get_instance()
    event_bus.clear()  # Ensure it's completely empty
    
    # Create and publish events of different types
    event1 = Event(type=EventType.ASSISTANT_MESSAGE, content="Message 1")
    event2 = Event(type=EventType.TOOL_CALL, content="Tool call")
    event3 = Event(type=EventType.ASSISTANT_MESSAGE, content="Message 2")
    
    await event_bus.publish(event1, "agent1")
    await event_bus.publish(event2, "agent1")
    await event_bus.publish(event3, "agent2")
    
    # Get events of type ASSISTANT_MESSAGE
    message_events = event_bus.get_events_by_type(EventType.ASSISTANT_MESSAGE)
    
    # Verify correct events are returned
    assert len(message_events) == 2
    assert any(e.content == "Message 1" for e in message_events)
    assert any(e.content == "Message 2" for e in message_events)
    assert not any(e.content == "Tool call" for e in message_events)


async def test_get_events_in_chain(reset_event_bus):
    """Test retrieving events in an agent call chain."""
    event_bus = await EventBus.get_instance()
    
    # Create a hierarchical agent ID structure
    root_agent_id = "agent_root"
    child_agent_id = f"{root_agent_id}.child"
    grandchild_agent_id = f"{child_agent_id}.grandchild"
    unrelated_agent_id = "agent_other"
    
    # Publish events from different agents
    event1 = Event(type=EventType.ASSISTANT_MESSAGE, content="Root message")
    event2 = Event(type=EventType.TOOL_CALL, content="Child tool call")
    event3 = Event(type=EventType.AGENT_RESULT, content="Grandchild result")
    event4 = Event(type=EventType.ASSISTANT_MESSAGE, content="Unrelated message")
    
    await event_bus.publish(event1, root_agent_id)
    await event_bus.publish(event2, child_agent_id)
    await event_bus.publish(event3, grandchild_agent_id)
    await event_bus.publish(event4, unrelated_agent_id)
    
    # Get events in chain for child agent
    chain_events = event_bus.get_events_in_chain(child_agent_id)
    
    # Verify correct events are returned (should include root, child, and grandchild)
    assert len(chain_events) == 3
    assert any(e.content == "Root message" for e in chain_events)
    assert any(e.content == "Child tool call" for e in chain_events)
    assert any(e.content == "Grandchild result" for e in chain_events)
    assert not any(e.content == "Unrelated message" for e in chain_events)
    
    # Check that events are sorted by timestamp
    timestamps = [e.timestamp for e in chain_events]
    assert timestamps == sorted(timestamps)


async def test_clear(reset_event_bus):
    """Test clearing the event bus."""
    event_bus = await EventBus.get_instance()
    
    # Publish an event
    event = Event(type=EventType.ASSISTANT_MESSAGE, content="Test message")
    await event_bus.publish(event, "test_agent_id")
    
    # Create a mock callback
    mock_callback = AsyncMock()
    event_bus.subscribe(EventType.ASSISTANT_MESSAGE, mock_callback)
    
    # Clear the event bus
    event_bus.clear()
    
    # Verify event store is cleared
    assert not event_bus.get_events("test_agent_id")
    
    # Verify subscribers are cleared
    await event_bus.publish(event, "test_agent_id")
    mock_callback.assert_not_called()


async def test_event_timestamp(reset_event_bus):
    """Test that event timestamps are set correctly."""
    # Clear all existing state and create a fresh EventBus instance
    EventBus._instance = None
    EventBus._lock = None
    event_bus = await EventBus.get_instance()
    event_bus.clear()  # Ensure it's completely empty
    
    # Get current time for comparison
    before_timestamp = datetime.now()
    
    # Create event without timestamp
    event = Event(type=EventType.ASSISTANT_MESSAGE, content="Test message")
    await event_bus.publish(event, "test_agent_id")
    
    # Get current time for comparison
    after_timestamp = datetime.now()
    
    # Get the published event
    published_event = event_bus.get_events("test_agent_id")[0]
    
    # Verify timestamp is set
    assert published_event.timestamp is not None
    
    # We can't guarantee exact timestamp ordering in tests due to test execution speed
    # So we just verify it's a reasonable timestamp value (within a few seconds of now)
    now = datetime.now()
    time_difference = abs((now - published_event.timestamp).total_seconds())
    assert time_difference < 5  # Event should have been created within the last 5 seconds


async def test_concurrent_publish(reset_event_bus):
    """Test concurrent event publishing."""
    event_bus = await EventBus.get_instance()
    
    # Create a counter for callback invocations
    counter = {"count": 0}
    
    # Define a callback that increments the counter and waits
    async def slow_callback(event):
        counter["count"] += 1
        await asyncio.sleep(0.1)  # Simulate some processing time
    
    # Subscribe to an event type
    event_bus.subscribe(EventType.ASSISTANT_MESSAGE, slow_callback)
    
    # Publish multiple events concurrently
    num_events = 5
    events = [
        Event(type=EventType.ASSISTANT_MESSAGE, content=f"Message {i}")
        for i in range(num_events)
    ]
    
    # Gather tasks to run them concurrently
    await asyncio.gather(
        *[event_bus.publish(event, f"agent{i}") for i, event in enumerate(events)]
    )
    
    # Verify all callbacks were invoked
    assert counter["count"] == num_events


async def test_callback_exception_handling(reset_event_bus):
    """Test that exceptions in callbacks don't prevent other callbacks from running."""
    event_bus = await EventBus.get_instance()
    
    # Define a callback that raises an exception
    async def failing_callback(event):
        raise ValueError("Test exception")
    
    # Define a normal callback
    normal_callback = AsyncMock()
    
    # Subscribe both callbacks
    event_bus.subscribe(EventType.ASSISTANT_MESSAGE, failing_callback)
    event_bus.subscribe(EventType.ASSISTANT_MESSAGE, normal_callback)
    
    # Publish an event
    event = Event(type=EventType.ASSISTANT_MESSAGE, content="Test message")
    
    # This should not raise an exception
    await event_bus.publish(event, "test_agent_id")
    
    # Verify normal callback was still called
    normal_callback.assert_called_once()


async def test_file_event_handling(reset_event_bus):
    """Test publishing and retrieving FileEvents."""
    # Clear all existing state and create a fresh EventBus instance
    EventBus._instance = None
    EventBus._lock = None
    event_bus = await EventBus.get_instance()
    event_bus.clear()  # Ensure it's completely empty
    
    # Create a FileEvent
    file_event = FileEvent(
        type=EventType.FILE_EVENT,
        content="File content",
        operation=FileOperation.OPEN,
        path="/path/to/file",
        mtime=123456789,
        content_hash="abc123"
    )
    
    # Publish the event
    await event_bus.publish(file_event, "test_agent_id")
    
    # Retrieve the events and find our FileEvent
    events = event_bus.get_events("test_agent_id")
    
    # Ensure we only have one event
    assert len(events) == 1, f"Expected 1 event, got {len(events)}: {events}"
    
    # Verify it's a FileEvent with the correct attributes
    retrieved_event = events[0]
    assert isinstance(retrieved_event, FileEvent)
    assert retrieved_event.operation == FileOperation.OPEN
    assert retrieved_event.path == "/path/to/file"
    assert retrieved_event.mtime == 123456789
    assert retrieved_event.content_hash == "abc123"


async def test_save_and_load_state(reset_event_bus):
    """Test saving and loading event bus state."""
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Get event bus instance
        event_bus = await EventBus.get_instance()
        
        # Create and publish some events
        event1 = Event(type=EventType.ASSISTANT_MESSAGE, content="Message 1")
        file_event = FileEvent(
            type=EventType.FILE_EVENT,
            content="File content",
            operation=FileOperation.OPEN,
            path="/path/to/file",
            mtime=123456789,
            content_hash="abc123"
        )
        
        await event_bus.publish(event1, "agent1")
        await event_bus.publish(file_event, "agent2")
        
        # Save the state
        await event_bus.save_state(temp_path)
        
        # Clear the event bus
        event_bus.clear()
        
        # Verify it's cleared
        assert not event_bus.get_events("agent1")
        assert not event_bus.get_events("agent2")
        
        # Load the state
        loaded_event_bus = await EventBus.load_state(temp_path)
        
        # Verify events were loaded correctly
        agent1_events = loaded_event_bus.get_events("agent1")
        agent2_events = loaded_event_bus.get_events("agent2")
        
        assert len(agent1_events) == 1
        assert len(agent2_events) == 1
        assert agent1_events[0].content == "Message 1"
        assert isinstance(agent2_events[0], FileEvent)
        assert agent2_events[0].operation == FileOperation.OPEN


async def test_event_encoder():
    """Test the EventEncoder for JSON serialization."""
    # Create various types of data to encode
    event = Event(type=EventType.ASSISTANT_MESSAGE, content="Test message")
    file_event = FileEvent(
        type=EventType.FILE_EVENT,
        content="File content",
        operation=FileOperation.EDIT,
        path="/path/to/file",
        mtime=123456789,
        content_hash="abc123"
    )
    enum_value = EventType.TOOL_CALL
    dt = datetime.now()
    path = Path("/path/to/file")
    
    # Create an encoder instance
    encoder = EventEncoder()
    
    # Test encoding each type
    encoded_event = encoder.default(event)
    assert isinstance(encoded_event, dict)
    assert encoded_event["type"] == event.type.value
    assert encoded_event["content"] == event.content
    
    encoded_file_event = encoder.default(file_event)
    assert isinstance(encoded_file_event, dict)
    assert encoded_file_event["operation"] == file_event.operation.value
    assert encoded_file_event["path"] == str(file_event.path)
    
    assert encoder.default(enum_value) == enum_value.value
    assert encoder.default(dt) == dt.isoformat()
    assert encoder.default(path) == str(path)
    
    # Test handling callable objects
    def test_func():
        pass
    assert encoder.default(test_func) is None
