"""
ML-TA Frontend Connector
This module connects the enhanced web frontend with the existing Flask API.
It serves the static frontend files and proxies API requests to the backend.
"""

import os
import sys
import logging
from flask import Flask, send_from_directory, request, Response
import requests

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ml-ta-frontend')

# Create Flask app for serving frontend
app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

# Configuration
class Config:
    # Backend API URL (default: localhost:8000)
    API_URL = os.environ.get('ML_TA_API_URL', 'http://localhost:8000')
    # Frontend port
    FRONTEND_PORT = int(os.environ.get('ML_TA_FRONTEND_PORT', 8080))
    # Debug mode
    DEBUG = os.environ.get('ML_TA_DEBUG', 'False').lower() == 'true'

# Serve static frontend files
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

# API proxy routes
@app.route('/api/v1/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_proxy(endpoint):
    """
    Proxy requests to the ML-TA backend API.
    This allows the frontend to make API calls without CORS issues.
    """
    # Build target URL
    url = f"{Config.API_URL}/api/v1/{endpoint}"
    logger.info(f"Proxying {request.method} request to {url}")
    
    # Forward headers, excluding host
    headers = {key: value for key, value in request.headers
               if key.lower() != 'host'}
    
    try:
        # Make the request to the backend API
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            params=request.args,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False
        )
        
        # Create response object
        response = Response(resp.content, resp.status_code)
        
        # Copy headers from backend response
        for key, value in resp.headers.items():
            if key.lower() != 'transfer-encoding':
                response.headers[key] = value
                
        return response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying request: {e}")
        return {"error": "Backend API service unavailable"}, 503

def run_frontend_server():
    """Run the frontend server."""
    logger.info(f"Starting ML-TA Frontend on port {Config.FRONTEND_PORT}")
    logger.info(f"Connecting to backend API at {Config.API_URL}")
    app.run(host='0.0.0.0', port=Config.FRONTEND_PORT, debug=Config.DEBUG)

if __name__ == '__main__':
    run_frontend_server()
