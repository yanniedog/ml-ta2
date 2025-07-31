"""Unit tests for the Flask API in api.py."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

# Mock Flask and other libraries before importing the API module
# This prevents errors if Flask is not installed in the test environment.
from src.api import create_api_server, AuthManager, RateLimiter, User


@pytest.fixture(scope="module")
def test_api_server():
    """Fixture to create a test API server instance."""
    # Use a mock for the model server to isolate API tests
    with patch("src.api.create_model_server") as mock_create_model_server:
        mock_model_server = MagicMock()
        mock_create_model_server.return_value = mock_model_server
        server = create_api_server(debug=True)
        yield server


@pytest.fixture(scope="module")
def client(test_api_server):
    """Fixture to provide a test client for the Flask app."""
    return test_api_server.get_app().test_client()


@pytest.fixture
def auth_manager(test_api_server):
    """Fixture to provide the AuthManager instance."""
    return test_api_server.auth_manager


# --------------------------------------------------------------------------
# AuthManager Tests
# --------------------------------------------------------------------------

def test_auth_manager_default_users(auth_manager: AuthManager):
    """Test that default admin and user are created."""
    assert len(auth_manager.users) >= 2
    admin_user = auth_manager.get_user("admin_001")
    test_user = auth_manager.get_user("user_001")
    assert admin_user is not None
    assert test_user is not None
    assert admin_user.role == "admin"
    assert test_user.role == "user"


def test_authenticate_api_key(auth_manager: AuthManager):
    """Test API key authentication."""
    admin_user = auth_manager.get_user("admin_001")
    authenticated_user = auth_manager.authenticate_api_key(admin_user.api_key)
    assert authenticated_user is not None
    assert authenticated_user.user_id == admin_user.user_id

    # Test invalid key
    invalid_user = auth_manager.authenticate_api_key("invalid_key")
    assert invalid_user is None


def test_jwt_generation_and_authentication(auth_manager: AuthManager):
    """Test JWT generation and validation."""
    user = auth_manager.get_user("user_001")
    token = auth_manager.generate_jwt(user)
    assert token is not None

    # Test valid token
    decoded_user, _ = auth_manager.authenticate_jwt(token)
    assert decoded_user is not None
    assert decoded_user.user_id == user.user_id

    # Test expired token
    expired_token = auth_manager.generate_jwt(user, expiration_delta_seconds=-1)
    expired_user, error = auth_manager.authenticate_jwt(expired_token)
    assert expired_user is None
    assert "expired" in error.lower()


# --------------------------------------------------------------------------
# RateLimiter Tests
# --------------------------------------------------------------------------

def test_rate_limiter():
    """Test the RateLimiter functionality."""
    limiter = RateLimiter(requests_per_minute=5)
    user_id = "test_rate_limit_user"

    # First 5 requests should be allowed
    for _ in range(5):
        assert limiter.is_allowed(user_id) is True

    # 6th request should be denied
    assert limiter.is_allowed(user_id) is False

    # Check remaining requests
    remaining = limiter.get_remaining_requests(user_id)
    assert remaining == 0


# --------------------------------------------------------------------------
# API Endpoint Tests
# --------------------------------------------------------------------------

def test_health_check_endpoint(client):
    """Test the /health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "ok"
    assert "model_server_status" in data


def test_predict_endpoint_unauthorized(client):
    """Test that the /predict endpoint requires authentication."""
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 401  # Unauthorized


def test_predict_endpoint_authorized(client, auth_manager, test_api_server):
    """Test a successful call to the /predict endpoint."""
    user = auth_manager.get_user("user_001")
    headers = {"X-API-Key": user.api_key}

    # Mock the prediction engine's response
    mock_engine = MagicMock()
    mock_engine.predict.return_value = {"prediction": 0.75, "confidence": 0.9}
    test_api_server.prediction_engine = mock_engine

    response = client.post("/api/v1/predict", headers=headers, json={"features": [1, 2, 3]})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["prediction"] == 0.75
    assert data["confidence"] == 0.9
    mock_engine.predict.assert_called_once()


def test_rate_limiting_on_endpoint(client, auth_manager):
    """Test that API endpoints are rate-limited."""
    user = auth_manager.get_user("user_001")
    headers = {"X-API-Key": user.api_key}

    # Exhaust the rate limit (default is 100, so we don't test all 100)
    # Instead, we'll patch the limiter to have a small limit
    with patch.object(RateLimiter, "__init__", return_value=None),
         patch.object(RateLimiter, "requests", {user.user_id: [time.time()] * 3}),
         patch.object(RateLimiter, "requests_per_minute", 3):

        # This request should be denied
        response = client.get("/api/v1/models", headers=headers)
        assert response.status_code == 429  # Too Many Requests
