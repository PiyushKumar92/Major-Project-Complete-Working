from flask import render_template, request, jsonify, flash, redirect, url_for
from __init__ import db
import logging

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    """Register global error handlers for the application"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        logger.warning(f"400 Bad Request: {request.url} - {str(error)}")
        
        if request.is_json:
            return jsonify({
                'error': 'Bad Request',
                'message': 'The request could not be understood by the server.',
                'status_code': 400
            }), 400
        
        return render_template('errors/400.html'), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors"""
        logger.warning(f"401 Unauthorized: {request.url} - {str(error)}")
        
        if request.is_json:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authentication required.',
                'status_code': 401
            }), 401
        
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('main.login'))
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors"""
        logger.warning(f"403 Forbidden: {request.url} - {str(error)}")
        
        if request.is_json:
            return jsonify({
                'error': 'Forbidden',
                'message': 'You do not have permission to access this resource.',
                'status_code': 403
            }), 403
        
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        logger.info(f"404 Not Found: {request.url}")
        
        if request.is_json:
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found.',
                'status_code': 404
            }), 404
        
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle 413 Request Entity Too Large errors"""
        logger.warning(f"413 Request Too Large: {request.url}")
        
        if request.is_json:
            return jsonify({
                'error': 'File Too Large',
                'message': 'The uploaded file is too large. Maximum size is 16MB.',
                'status_code': 413
            }), 413
        
        flash('File too large. Maximum upload size is 16MB.', 'error')
        return redirect(request.referrer or url_for('main.index'))
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors"""
        logger.error(f"500 Internal Server Error: {request.url} - {str(error)}")
        
        # Rollback database session in case of error
        try:
            db.session.rollback()
        except:
            pass
        
        if request.is_json:
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred. Please try again later.',
                'status_code': 500
            }), 500
        
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(502)
    def bad_gateway(error):
        """Handle 502 Bad Gateway errors"""
        logger.error(f"502 Bad Gateway: {request.url}")
        
        if request.is_json:
            return jsonify({
                'error': 'Bad Gateway',
                'message': 'The server received an invalid response from upstream.',
                'status_code': 502
            }), 502
        
        return render_template('errors/502.html'), 502
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable errors"""
        logger.error(f"503 Service Unavailable: {request.url}")
        
        if request.is_json:
            return jsonify({
                'error': 'Service Unavailable',
                'message': 'The service is temporarily unavailable. Please try again later.',
                'status_code': 503
            }), 503
        
        return render_template('errors/503.html'), 503
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle all other exceptions"""
        logger.error(f"Unhandled Exception: {request.url} - {str(error)}", exc_info=True)
        
        # Rollback database session
        try:
            db.session.rollback()
        except:
            pass
        
        if request.is_json:
            return jsonify({
                'error': 'Unexpected Error',
                'message': 'An unexpected error occurred. Please try again later.',
                'status_code': 500
            }), 500
        
        # For development, show the actual error
        if app.debug:
            raise error
        
        return render_template('errors/500.html'), 500
