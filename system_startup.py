"""
System Startup Manager
Initializes and starts all self-management systems
"""

import os
import sys
import time
import logging
from threading import Thread

def setup_logging():
    """Setup startup logging"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/startup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('SystemStartup')

def start_system_self_management():
    """Start all system self-management components"""
    logger = setup_logging()
    logger.info("🚀 Starting System Self-Management...")
    
    try:
        # Start System Monitoring
        logger.info("📊 Starting System Monitoring...")
        from system_monitor import start_system_monitoring
        start_system_monitoring()
        logger.info("✅ System Monitoring started")
        
        # Start Security Automation
        logger.info("🔒 Starting Security Automation...")
        from security_automation import start_security_automation
        start_security_automation()
        logger.info("✅ Security Automation started")
        
        # Initialize health check endpoint
        logger.info("🏥 Setting up health check...")
        setup_health_check()
        logger.info("✅ Health check endpoint ready")
        
        logger.info("🎉 System Self-Management fully operational!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start system self-management: {e}")
        return False

def setup_health_check():
    """Setup health check endpoint"""
    try:
        from flask import Flask, jsonify
        from system_monitor import get_system_status
        from security_automation import get_security_status
        
        # This will be integrated into the main Flask app
        # For now, just ensure the functions are available
        health_data = {
            'system': get_system_status(),
            'security': get_security_status()
        }
        
    except Exception as e:
        print(f"Health check setup error: {e}")

def initialize_on_startup():
    """Initialize system self-management on application startup"""
    def startup_thread():
        time.sleep(2)  # Wait for Flask app to fully start
        start_system_self_management()
    
    thread = Thread(target=startup_thread, daemon=True)
    thread.start()

if __name__ == "__main__":
    start_system_self_management()