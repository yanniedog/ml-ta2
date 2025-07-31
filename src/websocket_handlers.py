"""
WebSocket handlers for real-time ML-TA system communication.

This module implements:
- Real-time prediction streaming
- Live model performance monitoring
- Real-time alerts and notifications
- WebSocket authentication and connection management
- Bidirectional communication for admin interface
"""

import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# WebSocket libraries
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    # Mock websockets for testing
    class WebSocketServerProtocol:
        async def send(self, message): pass
        async def recv(self): return "{}"
        @property
        def closed(self): return False
    
    class websockets:
        @staticmethod
        async def serve(handler, host, port): pass

try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

import pandas as pd
import numpy as np

from src.config import get_config
from src.logging_config import get_logger
from src.prediction_engine import PredictionRequest, PredictionResponse
from src.api import AuthManager

logger = get_logger(__name__)


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    message_id: str
    user_id: Optional[str] = None


@dataclass
class ConnectionInfo:
    """WebSocket connection information."""
    connection_id: str
    user_id: str
    websocket: Any
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[str]
    is_authenticated: bool = False


class WebSocketManager:
    """Manage WebSocket connections and messaging."""
    
    def __init__(self, auth_manager: AuthManager):
        """Initialize WebSocket manager."""
        self.auth_manager = auth_manager
        self.connections: Dict[str, ConnectionInfo] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self.message_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.lock = threading.RLock()
        
        # Setup default message handlers
        self._setup_message_handlers()
        
        logger.info("WebSocketManager initialized")
    
    def _setup_message_handlers(self):
        """Setup default message handlers."""
        self.message_handlers.update({
            "auth": self._handle_auth_message,
            "subscribe": self._handle_subscribe_message,
            "unsubscribe": self._handle_unsubscribe_message,
            "predict": self._handle_predict_message,
            "ping": self._handle_ping_message
        })
    
    async def register_connection(self, websocket: WebSocketServerProtocol, 
                                connection_id: str) -> ConnectionInfo:
        """Register new WebSocket connection."""
        with self.lock:
            connection = ConnectionInfo(
                connection_id=connection_id,
                user_id="",  # Will be set during authentication
                websocket=websocket,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                subscriptions=set(),
                is_authenticated=False
            )
            
            self.connections[connection_id] = connection
            
            logger.info("WebSocket connection registered", connection_id=connection_id)
            
            return connection
    
    async def unregister_connection(self, connection_id: str):
        """Unregister WebSocket connection."""
        with self.lock:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                
                # Remove from all subscriptions
                for topic in connection.subscriptions:
                    if topic in self.subscriptions:
                        self.subscriptions[topic].discard(connection_id)
                        if not self.subscriptions[topic]:
                            del self.subscriptions[topic]
                
                del self.connections[connection_id]
                
                logger.info("WebSocket connection unregistered", 
                           connection_id=connection_id,
                           user_id=connection.user_id)
    
    async def authenticate_connection(self, connection_id: str, 
                                    auth_data: Dict[str, Any]) -> bool:
        """Authenticate WebSocket connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Try API key authentication
        api_key = auth_data.get("api_key")
        if api_key:
            user = self.auth_manager.authenticate_api_key(api_key)
            if user:
                connection.user_id = user.user_id
                connection.is_authenticated = True
                
                logger.info("WebSocket connection authenticated", 
                           connection_id=connection_id,
                           user_id=user.user_id)
                return True
        
        # Try JWT authentication
        token = auth_data.get("token")
        if token:
            user = self.auth_manager.authenticate_jwt(token)
            if user:
                connection.user_id = user.user_id
                connection.is_authenticated = True
                
                logger.info("WebSocket connection authenticated", 
                           connection_id=connection_id,
                           user_id=user.user_id)
                return True
        
        return False
    
    async def subscribe_to_topic(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to topic."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        if not connection.is_authenticated:
            return False
        
        with self.lock:
            connection.subscriptions.add(topic)
            
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            self.subscriptions[topic].add(connection_id)
            
            logger.debug("Connection subscribed to topic", 
                        connection_id=connection_id,
                        topic=topic)
            
            return True
    
    async def unsubscribe_from_topic(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe connection from topic."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        with self.lock:
            connection.subscriptions.discard(topic)
            
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(connection_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
            
            logger.debug("Connection unsubscribed from topic", 
                        connection_id=connection_id,
                        topic=topic)
            
            return True
    
    async def send_message(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        try:
            message_data = {
                "type": message.type,
                "data": message.data,
                "timestamp": message.timestamp.isoformat(),
                "message_id": message.message_id
            }
            
            if WEBSOCKETS_AVAILABLE:
                await connection.websocket.send(json.dumps(message_data))
            
            connection.last_activity = datetime.now()
            
            logger.debug("Message sent to connection", 
                        connection_id=connection_id,
                        message_type=message.type)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_topic(self, topic: str, message: WebSocketMessage):
        """Broadcast message to all subscribers of topic."""
        if topic not in self.subscriptions:
            return 0
        
        sent_count = 0
        failed_connections = []
        
        for connection_id in self.subscriptions[topic].copy():
            success = await self.send_message(connection_id, message)
            if success:
                sent_count += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.unregister_connection(connection_id)
        
        logger.debug("Message broadcasted to topic", 
                    topic=topic,
                    sent_count=sent_count,
                    failed_count=len(failed_connections))
        
        return sent_count
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """Handle incoming WebSocket message."""
        try:
            message_data = json.loads(raw_message)
            message_type = message_data.get("type")
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                await handler(connection_id, message_data)
            else:
                logger.warning("Unknown message type", 
                              connection_id=connection_id,
                              message_type=message_type)
                
                # Send error response
                error_message = WebSocketMessage(
                    type="error",
                    data={"error": f"Unknown message type: {message_type}"},
                    timestamp=datetime.now(),
                    message_id=f"error_{int(time.time() * 1000)}"
                )
                await self.send_message(connection_id, error_message)
            
            # Update last activity
            if connection_id in self.connections:
                self.connections[connection_id].last_activity = datetime.now()
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON message", connection_id=connection_id)
            
            error_message = WebSocketMessage(
                type="error",
                data={"error": "Invalid JSON format"},
                timestamp=datetime.now(),
                message_id=f"error_{int(time.time() * 1000)}"
            )
            await self.send_message(connection_id, error_message)
            
        except Exception as e:
            logger.error(f"Message handling error for {connection_id}: {e}")
    
    async def _handle_auth_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle authentication message."""
        auth_data = message_data.get("data", {})
        success = await self.authenticate_connection(connection_id, auth_data)
        
        response = WebSocketMessage(
            type="auth_response",
            data={
                "authenticated": success,
                "connection_id": connection_id
            },
            timestamp=datetime.now(),
            message_id=f"auth_resp_{int(time.time() * 1000)}"
        )
        
        await self.send_message(connection_id, response)
    
    async def _handle_subscribe_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle subscription message."""
        topic = message_data.get("data", {}).get("topic")
        
        if not topic:
            error_message = WebSocketMessage(
                type="error",
                data={"error": "Topic required for subscription"},
                timestamp=datetime.now(),
                message_id=f"error_{int(time.time() * 1000)}"
            )
            await self.send_message(connection_id, error_message)
            return
        
        success = await self.subscribe_to_topic(connection_id, topic)
        
        response = WebSocketMessage(
            type="subscribe_response",
            data={
                "topic": topic,
                "subscribed": success
            },
            timestamp=datetime.now(),
            message_id=f"sub_resp_{int(time.time() * 1000)}"
        )
        
        await self.send_message(connection_id, response)
    
    async def _handle_unsubscribe_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle unsubscription message."""
        topic = message_data.get("data", {}).get("topic")
        
        if not topic:
            error_message = WebSocketMessage(
                type="error",
                data={"error": "Topic required for unsubscription"},
                timestamp=datetime.now(),
                message_id=f"error_{int(time.time() * 1000)}"
            )
            await self.send_message(connection_id, error_message)
            return
        
        success = await self.unsubscribe_from_topic(connection_id, topic)
        
        response = WebSocketMessage(
            type="unsubscribe_response",
            data={
                "topic": topic,
                "unsubscribed": success
            },
            timestamp=datetime.now(),
            message_id=f"unsub_resp_{int(time.time() * 1000)}"
        )
        
        await self.send_message(connection_id, response)
    
    async def _handle_predict_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle prediction request message."""
        # This would integrate with the prediction engine
        # For now, send a mock response
        
        prediction_data = message_data.get("data", {})
        
        response = WebSocketMessage(
            type="prediction_response",
            data={
                "request_id": prediction_data.get("request_id", "unknown"),
                "predictions": [0, 1, 1],  # Mock predictions
                "processing_time_ms": 15.5,
                "model_name": prediction_data.get("model_name", "default")
            },
            timestamp=datetime.now(),
            message_id=f"pred_resp_{int(time.time() * 1000)}"
        )
        
        await self.send_message(connection_id, response)
    
    async def _handle_ping_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle ping message."""
        response = WebSocketMessage(
            type="pong",
            data={"timestamp": datetime.now().isoformat()},
            timestamp=datetime.now(),
            message_id=f"pong_{int(time.time() * 1000)}"
        )
        
        await self.send_message(connection_id, response)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        with self.lock:
            total_connections = len(self.connections)
            authenticated_connections = sum(1 for conn in self.connections.values() 
                                          if conn.is_authenticated)
            
            topic_stats = {}
            for topic, connection_ids in self.subscriptions.items():
                topic_stats[topic] = len(connection_ids)
            
            return {
                "total_connections": total_connections,
                "authenticated_connections": authenticated_connections,
                "unauthenticated_connections": total_connections - authenticated_connections,
                "topic_subscriptions": topic_stats,
                "total_topics": len(self.subscriptions)
            }


class RealtimeMonitor:
    """Real-time monitoring and alerting via WebSocket."""
    
    def __init__(self, websocket_manager: WebSocketManager):
        """Initialize real-time monitor."""
        self.websocket_manager = websocket_manager
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_history = []
        self.alert_thresholds = {
            "processing_time_ms": 100,
            "error_rate_percent": 5,
            "memory_usage_percent": 80
        }
        
        logger.info("RealtimeMonitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        if ASYNCIO_AVAILABLE:
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Check for alerts
                alerts = self._check_alerts(metrics)
                
                # Broadcast metrics to subscribers
                asyncio.run(self._broadcast_metrics(metrics))
                
                # Broadcast alerts if any
                if alerts:
                    asyncio.run(self._broadcast_alerts(alerts))
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        # Mock metrics collection
        current_time = datetime.now()
        
        metrics = {
            "timestamp": current_time.isoformat(),
            "processing_time_ms": np.random.uniform(10, 50),
            "error_rate_percent": np.random.uniform(0, 2),
            "memory_usage_percent": np.random.uniform(30, 70),
            "active_connections": len(self.websocket_manager.connections),
            "predictions_per_second": np.random.uniform(10, 100),
            "models_deployed": np.random.randint(1, 5)
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 metrics
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if value > threshold:
                    alert = {
                        "type": "threshold_exceeded",
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "severity": "warning" if value < threshold * 1.5 else "critical",
                        "timestamp": datetime.now().isoformat()
                    }
                    alerts.append(alert)
        
        return alerts
    
    async def _broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast metrics to subscribers."""
        message = WebSocketMessage(
            type="metrics_update",
            data=metrics,
            timestamp=datetime.now(),
            message_id=f"metrics_{int(time.time() * 1000)}"
        )
        
        await self.websocket_manager.broadcast_to_topic("metrics", message)
    
    async def _broadcast_alerts(self, alerts: List[Dict[str, Any]]):
        """Broadcast alerts to subscribers."""
        for alert in alerts:
            message = WebSocketMessage(
                type="alert",
                data=alert,
                timestamp=datetime.now(),
                message_id=f"alert_{int(time.time() * 1000)}"
            )
            
            await self.websocket_manager.broadcast_to_topic("alerts", message)
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics history."""
        return self.metrics_history[-limit:]


class WebSocketServer:
    """WebSocket server for ML-TA system."""
    
    def __init__(self, auth_manager: AuthManager, host: str = "localhost", port: int = 8081):
        """Initialize WebSocket server."""
        self.host = host
        self.port = port
        self.auth_manager = auth_manager
        self.websocket_manager = WebSocketManager(auth_manager)
        self.realtime_monitor = RealtimeMonitor(self.websocket_manager)
        self.server = None
        self.is_running = False
        
        logger.info("WebSocketServer initialized", host=host, port=port)
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        connection_id = f"conn_{int(time.time() * 1000)}_{id(websocket)}"
        
        try:
            # Register connection
            await self.websocket_manager.register_connection(websocket, connection_id)
            
            logger.info("WebSocket connection established", 
                       connection_id=connection_id,
                       path=path)
            
            # Handle messages
            async for message in websocket:
                await self.websocket_manager.handle_message(connection_id, message)
                
        except Exception as e:
            logger.error(f"WebSocket connection error for {connection_id}: {e}")
            
        finally:
            # Unregister connection
            await self.websocket_manager.unregister_connection(connection_id)
            
            logger.info("WebSocket connection closed", connection_id=connection_id)
    
    async def start_async(self):
        """Start WebSocket server (async)."""
        if WEBSOCKETS_AVAILABLE:
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port
            )
            
            logger.info("WebSocket server started", host=self.host, port=self.port)
        else:
            logger.info("WebSockets not available - server in mock mode")
        
        # Start real-time monitoring
        self.realtime_monitor.start_monitoring()
        
        self.is_running = True
    
    def start(self):
        """Start WebSocket server (sync wrapper)."""
        if ASYNCIO_AVAILABLE:
            asyncio.run(self.start_async())
        else:
            logger.info("AsyncIO not available - WebSocket server in mock mode")
            self.is_running = True
    
    def stop(self):
        """Stop WebSocket server."""
        self.is_running = False
        
        # Stop monitoring
        self.realtime_monitor.stop_monitoring()
        
        if self.server:
            self.server.close()
        
        logger.info("WebSocket server stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return {
            "is_running": self.is_running,
            "host": self.host,
            "port": self.port,
            "connection_stats": self.websocket_manager.get_connection_stats(),
            "monitoring_active": self.realtime_monitor.monitoring_active
        }


def create_websocket_server(auth_manager: AuthManager,
                           host: str = "localhost",
                           port: int = 8081) -> WebSocketServer:
    """Factory function to create WebSocket server."""
    return WebSocketServer(auth_manager, host, port)
