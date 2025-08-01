#!/usr/bin/env python3
"""
Main entry point for the ML-TA system.
This script starts both the API server and the web frontend.
"""

import os
import sys
import subprocess
import threading
import time
import signal
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api import create_api_server

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ml-ta-runner')


def run_api_server():
    """Run the ML-TA API server."""
    try:
        logger.info("Starting ML-TA API server on port 8000...")
        api_server = create_api_server(host="0.0.0.0", port=8000, debug=True)
        api_server.start()
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise


def run_frontend_server():
    """Run the ML-TA web frontend server."""
    try:
        logger.info("Starting ML-TA web frontend on port 8080...")
        # Import the frontend connector
        import web_app.connector as frontend
        frontend.run_frontend_server()
    except Exception as e:
        logger.error(f"Failed to start frontend server: {e}")
        raise


def main():
    """Main function to start both servers."""
    logger.info("Starting ML-TA system...")
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    # Start frontend server in main thread
    try:
        run_frontend_server()
    except KeyboardInterrupt:
        logger.info("Shutting down ML-TA system...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running ML-TA system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
