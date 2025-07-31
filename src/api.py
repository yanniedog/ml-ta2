"""
REST API for ML-TA system.

This module implements:
- REST API endpoints for predictions and model management
- Authentication and authorization
- Rate limiting and request validation
- API documentation and error handling
- Integration with prediction engine and model serving
"""

import json
import time
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core libraries
try:
    from flask import Flask, request, jsonify, g
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    # Mock Flask for testing
    class Flask:
        def __init__(self, name): pass
        def route(self, *args, **kwargs): 
            def decorator(func): return func
            return decorator
        def run(self, *args, **kwargs): pass
    
    class CORS:
        def __init__(self, app): pass
    
    def jsonify(data): return {"data": data}
    
    class request:
        @staticmethod
        def get_json(): return {}
        @staticmethod
        def get_header(name): return None
        args = {}
    
    class g:
        pass

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    # Mock JWT for testing
    class jwt:
        @staticmethod
        def encode(payload, secret, algorithm="HS256"):
            return "mock_token"
        
        @staticmethod
        def decode(token, secret, algorithms=None):
            return {"user_id": "test_user", "exp": time.time() + 3600}

import pandas as pd
import numpy as np

from src.config import get_config
from src.logging_config import get_logger
from src.prediction_engine import create_prediction_engine, PredictionRequest
from src.model_serving import create_model_server

logger = get_logger(__name__)


@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    secret_key: str = "ml-ta-secret-key"
    jwt_expiration_hours: int = 24
    rate_limit_requests_per_minute: int = 100
    max_request_size_mb: int = 10
    enable_cors: bool = True
    api_version: str = "v1"


@dataclass
class User:
    """User model for authentication."""
    user_id: str
    username: str
    email: str
    api_key: str
    role: str = "user"  # user, admin
    created_at: datetime = None
    last_login: datetime = None
    is_active: bool = True


class AuthManager:
    """Manage authentication and authorization."""
    
    def __init__(self, secret_key: str):
        """Initialize auth manager."""
        self.secret_key = secret_key
        self.users = {}  # In production, use database
        self.api_keys = {}
        self.sessions = {}
        
        # Create default admin user
        self._create_default_users()
        
        logger.info("AuthManager initialized")
    
    def _create_default_users(self):
        """Create default users for testing."""
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@ml-ta.com",
            api_key=self._generate_api_key(),
            role="admin",
            created_at=datetime.now()
        )
        
        test_user = User(
            user_id="user_001",
            username="testuser",
            email="test@ml-ta.com",
            api_key=self._generate_api_key(),
            role="user",
            created_at=datetime.now()
        )
        
        self.users[admin_user.user_id] = admin_user
        self.users[test_user.user_id] = test_user
        self.api_keys[admin_user.api_key] = admin_user.user_id
        self.api_keys[test_user.api_key] = test_user.user_id
        
        logger.info("Default users created", 
                   admin_api_key=admin_user.api_key[:8] + "...",
                   user_api_key=test_user.api_key[:8] + "...")
    
    def _generate_api_key(self) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(32)
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key."""
        if api_key in self.api_keys:
            user_id = self.api_keys[api_key]
            user = self.users.get(user_id)
            if user and user.is_active:
                user.last_login = datetime.now()
                return user
        return None
    
    def authenticate_jwt(self, token: str) -> Optional[User]:
        """Authenticate using JWT token."""
        try:
            if JWT_AVAILABLE:
                payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
                user_id = payload.get("user_id")
                
                if user_id in self.users:
                    user = self.users[user_id]
                    if user.is_active:
                        return user
            else:
                # Mock JWT authentication for testing
                if token == "mock_token":
                    return self.users.get("user_001")
                    
        except Exception as e:
            logger.warning(f"JWT authentication failed: {e}")
        
        return None
    
    def generate_jwt(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        if JWT_AVAILABLE:
            return jwt.encode(payload, self.secret_key, algorithm="HS256")
        else:
            return "mock_token"
    
    def create_user(self, username: str, email: str, role: str = "user") -> User:
        """Create new user."""
        user_id = f"user_{int(time.time())}"
        api_key = self._generate_api_key()
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            api_key=api_key,
            role=role,
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        self.api_keys[api_key] = user_id
        
        logger.info("User created", user_id=user_id, username=username, role=role)
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def list_users(self) -> List[User]:
        """List all users."""
        return list(self.users.values())


class RateLimiter:
    """Rate limiting for API requests."""
    
    def __init__(self, requests_per_minute: int = 100):
        """Initialize rate limiter."""
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # user_id -> list of timestamps
        
        logger.info("RateLimiter initialized", requests_per_minute=requests_per_minute)
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed for user."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                timestamp for timestamp in self.requests[user_id]
                if timestamp > minute_ago
            ]
        else:
            self.requests[user_id] = []
        
        # Check rate limit
        if len(self.requests[user_id]) >= self.requests_per_minute:
            return False
        
        # Record request
        self.requests[user_id].append(now)
        return True
    
    def get_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user."""
        if user_id not in self.requests:
            return self.requests_per_minute
        
        return max(0, self.requests_per_minute - len(self.requests[user_id]))


class APIValidator:
    """Validate API requests."""
    
    @staticmethod
    def validate_prediction_request(data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate prediction request data."""
        if not isinstance(data, dict):
            return False, "Request data must be a JSON object"
        
        if "features" not in data:
            return False, "Missing required field: features"
        
        features = data["features"]
        if not isinstance(features, (list, dict)):
            return False, "Features must be a list or dictionary"
        
        if isinstance(features, list) and len(features) == 0:
            return False, "Features list cannot be empty"
        
        if isinstance(features, dict) and len(features) == 0:
            return False, "Features dictionary cannot be empty"
        
        # Validate model_name if provided
        if "model_name" in data:
            model_name = data["model_name"]
            if not isinstance(model_name, str) or len(model_name.strip()) == 0:
                return False, "Model name must be a non-empty string"
        
        return True, "Valid"
    
    @staticmethod
    def validate_model_deployment_request(data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate model deployment request."""
        required_fields = ["model_name", "model_version"]
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
            
            if not isinstance(data[field], str) or len(data[field].strip()) == 0:
                return False, f"{field} must be a non-empty string"
        
        return True, "Valid"


class MLTA_API:
    """Main ML-TA API application."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize API application."""
        self.config = config or APIConfig()
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = self.config.secret_key
        
        # Initialize components
        self.auth_manager = AuthManager(self.config.secret_key)
        self.rate_limiter = RateLimiter(self.config.rate_limit_requests_per_minute)
        self.validator = APIValidator()
        
        # Initialize ML components
        self.prediction_engine = create_prediction_engine(
            max_latency_ms=100.0,
            enable_monitoring=True,
            enable_ab_testing=True
        )
        self.model_server = create_model_server()
        
        # Enable CORS if configured
        if self.config.enable_cors and FLASK_AVAILABLE:
            CORS(self.app)
        
        # Setup routes
        self._setup_routes()
        
        logger.info("MLTA_API initialized", 
                   host=self.config.host,
                   port=self.config.port,
                   api_version=self.config.api_version)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Authentication decorator
        def require_auth(f):
            def decorated_function(*args, **kwargs):
                # Check API key
                api_key = request.headers.get('X-API-Key')
                if api_key:
                    user = self.auth_manager.authenticate_api_key(api_key)
                    if user:
                        g.current_user = user
                        return f(*args, **kwargs)
                
                # Check JWT token
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    user = self.auth_manager.authenticate_jwt(token)
                    if user:
                        g.current_user = user
                        return f(*args, **kwargs)
                
                return jsonify({"error": "Authentication required"}), 401
            
            decorated_function.__name__ = f.__name__
            return decorated_function
        
        # Rate limiting decorator
        def require_rate_limit(f):
            def decorated_function(*args, **kwargs):
                if hasattr(g, 'current_user'):
                    if not self.rate_limiter.is_allowed(g.current_user.user_id):
                        return jsonify({
                            "error": "Rate limit exceeded",
                            "remaining_requests": self.rate_limiter.get_remaining_requests(g.current_user.user_id)
                        }), 429
                return f(*args, **kwargs)
            
            decorated_function.__name__ = f.__name__
            return decorated_function
        
        # Health check endpoint
        @self.app.route(f'/api/{self.config.api_version}/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": self.config.api_version,
                "components": {
                    "prediction_engine": "operational",
                    "model_server": "operational",
                    "auth_manager": "operational"
                }
            })
        
        # Authentication endpoints
        @self.app.route(f'/api/{self.config.api_version}/auth/login', methods=['POST'])
        def login():
            """Login endpoint."""
            data = request.get_json() or {}
            api_key = data.get('api_key')
            
            if not api_key:
                return jsonify({"error": "API key required"}), 400
            
            user = self.auth_manager.authenticate_api_key(api_key)
            if not user:
                return jsonify({"error": "Invalid API key"}), 401
            
            token = self.auth_manager.generate_jwt(user)
            
            return jsonify({
                "token": token,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "role": user.role
                },
                "expires_in": self.config.jwt_expiration_hours * 3600
            })
        
        # Prediction endpoints
        @self.app.route(f'/api/{self.config.api_version}/predict', methods=['POST'])
        @require_auth
        @require_rate_limit
        def predict():
            """Make prediction."""
            try:
                data = request.get_json() or {}
                
                # Validate request
                is_valid, message = self.validator.validate_prediction_request(data)
                if not is_valid:
                    return jsonify({"error": message}), 400
                
                # Convert features to DataFrame
                features_data = data["features"]
                if isinstance(features_data, list):
                    # Assume list of dictionaries
                    features_df = pd.DataFrame(features_data)
                elif isinstance(features_data, dict):
                    # Single prediction
                    features_df = pd.DataFrame([features_data])
                else:
                    return jsonify({"error": "Invalid features format"}), 400
                
                # Create prediction request
                prediction_request = PredictionRequest(
                    request_id=f"api_{int(time.time() * 1000)}",
                    timestamp=datetime.now(),
                    features=features_df,
                    model_name=data.get("model_name"),
                    metadata={"user_id": g.current_user.user_id}
                )
                
                # Make prediction
                response = self.prediction_engine.predict(prediction_request)
                
                # Format response
                result = {
                    "request_id": response.request_id,
                    "predictions": response.predictions.tolist() if hasattr(response.predictions, 'tolist') else response.predictions,
                    "probabilities": response.probabilities.tolist() if response.probabilities is not None else None,
                    "confidence_scores": response.confidence_scores.tolist() if response.confidence_scores is not None else None,
                    "model_name": response.model_name,
                    "processing_time_ms": response.processing_time_ms,
                    "timestamp": response.timestamp.isoformat()
                }
                
                if response.metadata and 'error' in response.metadata:
                    result["error"] = response.metadata["error"]
                    return jsonify(result), 500
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Prediction API error: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        # Model management endpoints
        @self.app.route(f'/api/{self.config.api_version}/models', methods=['GET'])
        @require_auth
        def list_models():
            """List available models."""
            try:
                status = self.model_server.get_model_status()
                models = status.get("models", []) if isinstance(status, dict) else []
                
                return jsonify({
                    "models": models,
                    "total_count": len(models)
                })
                
            except Exception as e:
                logger.error(f"List models API error: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route(f'/api/{self.config.api_version}/models/<model_name>/status', methods=['GET'])
        @require_auth
        def get_model_status(model_name):
            """Get model status."""
            try:
                status = self.model_server.get_model_status(model_name)
                
                if not status:
                    return jsonify({"error": "Model not found"}), 404
                
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Model status API error: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        # Admin endpoints
        @self.app.route(f'/api/{self.config.api_version}/admin/users', methods=['GET'])
        @require_auth
        def list_users():
            """List users (admin only)."""
            if g.current_user.role != "admin":
                return jsonify({"error": "Admin access required"}), 403
            
            try:
                users = self.auth_manager.list_users()
                user_data = []
                
                for user in users:
                    user_data.append({
                        "user_id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                        "role": user.role,
                        "created_at": user.created_at.isoformat() if user.created_at else None,
                        "last_login": user.last_login.isoformat() if user.last_login else None,
                        "is_active": user.is_active
                    })
                
                return jsonify({
                    "users": user_data,
                    "total_count": len(user_data)
                })
                
            except Exception as e:
                logger.error(f"List users API error: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route(f'/api/{self.config.api_version}/admin/metrics', methods=['GET'])
        @require_auth
        def get_metrics():
            """Get system metrics (admin only)."""
            if g.current_user.role != "admin":
                return jsonify({"error": "Admin access required"}), 403
            
            try:
                # Get prediction engine status
                engine_status = self.prediction_engine.get_status()
                
                # Get model server metrics
                server_metrics = self.model_server.get_metrics()
                
                return jsonify({
                    "prediction_engine": engine_status,
                    "model_server": server_metrics,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Metrics API error: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        # API documentation endpoint
        @self.app.route(f'/api/{self.config.api_version}/docs', methods=['GET'])
        def api_docs():
            """API documentation."""
            docs = {
                "title": "ML-TA API Documentation",
                "version": self.config.api_version,
                "base_url": f"/api/{self.config.api_version}",
                "authentication": {
                    "methods": ["API Key", "JWT Token"],
                    "api_key_header": "X-API-Key",
                    "jwt_header": "Authorization: Bearer <token>"
                },
                "endpoints": {
                    "health": {
                        "method": "GET",
                        "path": "/health",
                        "description": "Health check endpoint",
                        "authentication": False
                    },
                    "login": {
                        "method": "POST",
                        "path": "/auth/login",
                        "description": "Authenticate and get JWT token",
                        "authentication": False,
                        "body": {"api_key": "string"}
                    },
                    "predict": {
                        "method": "POST",
                        "path": "/predict",
                        "description": "Make predictions",
                        "authentication": True,
                        "body": {
                            "features": "array|object",
                            "model_name": "string (optional)"
                        }
                    },
                    "list_models": {
                        "method": "GET",
                        "path": "/models",
                        "description": "List available models",
                        "authentication": True
                    },
                    "model_status": {
                        "method": "GET",
                        "path": "/models/{model_name}/status",
                        "description": "Get model status",
                        "authentication": True
                    }
                },
                "rate_limits": {
                    "requests_per_minute": self.config.rate_limit_requests_per_minute
                }
            }
            
            return jsonify(docs)
    
    def start(self):
        """Start the API server."""
        try:
            # Start model server
            self.model_server.start()
            
            logger.info("Starting API server",
                       host=self.config.host,
                       port=self.config.port,
                       debug=self.config.debug)
            
            if FLASK_AVAILABLE:
                self.app.run(
                    host=self.config.host,
                    port=self.config.port,
                    debug=self.config.debug
                )
            else:
                logger.info("Flask not available - API server in mock mode")
                
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise
    
    def stop(self):
        """Stop the API server."""
        try:
            self.model_server.stop()
            logger.info("API server stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop API server: {e}")
    
    def get_app(self):
        """Get Flask app instance for testing."""
        return self.app


def create_api_server(host: str = "0.0.0.0", 
                     port: int = 8080,
                     debug: bool = False,
                     enable_cors: bool = True) -> MLTA_API:
    """Factory function to create API server."""
    config = APIConfig(
        host=host,
        port=port,
        debug=debug,
        enable_cors=enable_cors,
        secret_key=secrets.token_urlsafe(32),
        jwt_expiration_hours=24,
        rate_limit_requests_per_minute=100,
        max_request_size_mb=10,
        api_version="v1"
    )
    
    return MLTA_API(config)
