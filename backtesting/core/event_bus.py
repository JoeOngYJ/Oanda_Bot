# backtesting/core/event_bus.py

from typing import Callable, Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class EventType(Enum):
    """Event types in the backtest lifecycle"""
    BAR_CLOSED = "bar_closed"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_FINISHED = "backtest_finished"

@dataclass
class Event:
    """Generic event container"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]

class EventBus:
    """Simple pub/sub event bus for backtest components"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def publish(self, event: Event):
        """Publish an event to all subscribers"""
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                callback(event)