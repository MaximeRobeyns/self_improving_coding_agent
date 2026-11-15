# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""FastAPI server for callgraph visualization with enhanced formatting."""

import json
import asyncio
import logging

from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..callgraph.manager import CallGraphManager
from ..callgraph.reporting import _format_duration, _format_cost
from ..events import EventBus
from ..events.event_bus_utils import get_subagent_events
from ..types.event_types import EventType, Event, FileEvent

logger = logging.getLogger(__name__)


# Models
class NodeData(BaseModel):
    """Node data for visualization."""

    id: str
    name: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    token_count: Optional[int]
    num_cached_tokens: Optional[int]
    cost: Optional[float]
    success: Optional[bool]
    events: List[Dict]
    children: List[str]


class CallGraphData(BaseModel):
    """Complete callgraph data for visualization."""

    nodes: Dict[str, NodeData]
    root_id: Optional[str]
    total_duration: Optional[float]
    total_tokens: Optional[int]
    num_cached_tokens: Optional[int]
    total_cost: Optional[float]


# Server setup
app = FastAPI(title="Callgraph Visualizer")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

# Store events by node ID
event_store: Dict[str, List[Event | FileEvent]] = {}
active_websockets: Set[WebSocket] = set()


def format_event(event: Event | FileEvent) -> Dict:
    """Format an event for visualization."""
    try:
        # Handle Completion objects specifically
        metadata = event.metadata
        for k, v in metadata.items():
            if hasattr(v, "model_dump_json"):  # Check if it's a Pydantic model
                metadata[k] = json.loads(v.model_dump_json())
            elif not isinstance(v, (str, dict, list, int, float, bool, type(None))):
                metadata[k] = str(v)  # Convert non-serializable objects to string

        base_data = {
            "type": event.type.value,
            "content": str(event.content).strip() if event.content else "",
            "timestamp": (
                event.timestamp.isoformat()
                if event.timestamp
                else datetime.now().isoformat()
            ),
            "metadata": metadata if metadata else {},
        }

        if isinstance(event, FileEvent):
            base_data.update(
                {
                    "path": str(event.path),
                    "operation": (
                        event.operation.value
                        if hasattr(event.operation, "value")
                        else str(event.operation)
                    ),
                }
            )

        return base_data
    except Exception as e:
        logger.error(f"Error formatting event: {e}")
        return {
            "type": "error",
            "content": f"Error formatting event: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "metadata": {},
        }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the visualization UI."""
    return templates.TemplateResponse(request, "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time updates."""
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.remove(websocket)


async def broadcast_event(event: Event | FileEvent, node_id: str):
    """Broadcast an event to all connected clients."""
    formatted_event = format_event(event)
    message = {"type": "event", "nodeId": node_id, "event": formatted_event}
    websockets_copy = set(active_websockets)
    for websocket in websockets_copy:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting to websocket: {e}")
            active_websockets.remove(websocket)


@app.get("/api/callgraph", response_model=CallGraphData)
async def get_callgraph():
    """Get current callgraph data formatted for visualization."""
    manager = await CallGraphManager.get_instance()
    graph = manager.graph

    if not graph.root:
        return CallGraphData(
            nodes={},
            root_id=None,
            total_duration=None,
            total_tokens=None,
            num_cached_tokens=None,
            total_cost=None,
        )

    # Build nodes data
    nodes = {}
    for node_id, node in graph.nodes.items():
        try:
            # Get events for this node from event bus
            events = await get_subagent_events(node_id, set(EventType))

            nodes[node_id] = NodeData(
                id=node_id,
                name=node.name,
                started_at=node.started_at,
                completed_at=node.completed_at,
                duration_seconds=node.duration_seconds,
                token_count=node.token_count,
                num_cached_tokens=node.num_cached_tokens,
                cost=node.cost,
                success=node.success,
                events=[format_event(event) for event in events],
                children=sorted(list(node.children)),
            )
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {e}")
            continue

    # Get metrics
    metrics = graph.get_execution_metrics()

    return CallGraphData(
        nodes=nodes,
        root_id=graph.root.id,
        total_duration=metrics["total_duration"],
        total_tokens=metrics["total_tokens"],
        num_cached_tokens=metrics["num_cached_tokens"],
        total_cost=metrics["total_cost"],
    )


async def event_callback(event: Event | FileEvent):
    """Handle new events from EventBus."""
    publisher_id = event.metadata.get("publisher_id")
    if publisher_id:
        if publisher_id not in event_store:
            event_store[publisher_id] = []
        event_store[publisher_id].append(event)
        await broadcast_event(event, publisher_id)


@app.on_event("startup")
async def startup_event():
    """Initialize event subscriptions on server startup."""
    event_bus = await EventBus.get_instance()
    # Subscribe to all relevant event types
    event_bus.subscribe(set(EventType), event_callback)


@app.on_event("shutdown")
async def shutdown_event():
    """Handle graceful server shutdown."""
    logger.info("Shutting down web visualization server...")
    event_bus = await EventBus.get_instance()
    # Unsubscribe from all event types
    for event_type in EventType:
        event_bus.unsubscribe(event_type, event_callback)


class UvicornServer(uvicorn.Server):
    """Customized uvicorn server with graceful shutdown."""

    async def startup(self, sockets: Optional[List] = None) -> None:
        """
        Override startup to handle it more gracefully.
        """
        try:
            await super().startup(sockets)
        except Exception as e:
            logger.warning(f"Error during server startup: {e}")

    async def shutdown(self, sockets: Optional[List] = None) -> None:
        """
        Override shutdown to handle it more gracefully.
        """
        try:
            await super().shutdown(sockets)
        except Exception as e:
            logger.warning(f"Error during server shutdown: {e}")


async def run_server():
    """Run the FastAPI server using uvicorn with graceful shutdown."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="error")
    server = UvicornServer(config=config)

    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Web server task cancelled, shutting down gracefully...")
        await server.shutdown()
    except Exception as e:
        logger.error(f"Error running web server: {e}")
        raise
