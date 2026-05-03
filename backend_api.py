"""
COTCAgent Backend API
Provides RESTful API interfaces, connecting frontend with COTCAgent core functionality
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import asyncio
import json
import os
import logging
from cotc_agent import COTCAgent
from config_manager import ConfigManager, ConfigurationError
from performance_monitor import get_performance_stats, cleanup_memory

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variable to store agent instance
agent = None

def initialize_agent():
    """Initialize COTCAgent instance with configuration management"""
    global agent
    if agent is None:
        try:
            # Load configuration from environment variables and config file
            config_manager = ConfigManager()
            config = config_manager.get_deepseek_config()

            # Override with environment variable if set (for backward compatibility)
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key:
                config.api_key = api_key

            agent = COTCAgent(config)
            logging.info("COTCAgent initialized successfully with configuration management")

        except ConfigurationError as e:
            logging.error(f"Configuration error during agent initialization: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during agent initialization: {e}")
            raise
    return agent

@app.route('/')
def index():
    """Home page - Returns frontend interface"""
    return render_template('web_interface.html')

@app.route('/api/patient/info', methods=['GET'])
def get_patient_info():
    """Retrieve patient information"""
    try:
        # Load patient data
        with open('patient_data/patient_0001.json', 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
        
        return jsonify({
            'success': True,
            'data': {
                'patient_id': patient_data['patient_info']['id'],
                'total_indicators': patient_data['patient_info']['total_indicators'],
                'existing_diseases': len(patient_data['patient_info']['diseases']),
                'diseases': patient_data['patient_info']['diseases']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analysis/query', methods=['POST'])
def analyze_query():
    """Analyze user query with medical data"""
    try:
        data = request.get_json()
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Query content cannot be empty'
            }), 400

        # Initialize agent
        agent = initialize_agent()

        # Load patient data
        with open('patient_data/patient_0001.json', 'r', encoding='utf-8') as f:
            patient_data = json.load(f)

        # Process query asynchronously
        try:
            # Use asyncio.run() for cleaner async handling
            result = asyncio.run(agent.process_user_query(user_query, patient_data))
        except RuntimeError as e:
            # If we're already in an event loop, create a new one
            if 'already running' in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        agent.process_user_query(user_query, patient_data)
                    )
                finally:
                    loop.close()
            else:
                raise

        # Format response results
        formatted_result = {
            'success': True,
            'data': {
                'temporal_analysis': result.get('temporal_analysis', {}),
                'detailed_analysis': result.get('detailed_analysis', {}),
                'disease_risks': [
                    {
                        'disease_id': risk.disease_id,
                        'disease_name': risk.disease_name,
                        'risk_score': risk.risk_score,
                        'confidence': risk.confidence,
                        'matched_symptoms': risk.matched_symptoms,
                        'missing_symptoms': risk.missing_symptoms
                    }
                    for risk in result.get('disease_risks', [])
                ],
                'inquiry_questions': result.get('active_inquiry_questions', []),
                'comprehensive_analysis': result.get('comprehensive_analysis', {})
            }
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error occurred during analysis: {str(e)}'
        }), 500

@app.route('/api/analysis/status', methods=['GET'])
def get_analysis_status():
    """Get analysis status"""
    return jsonify({
        'success': True,
        'data': {
            'status': 'ready',
            'agent_initialized': agent is not None,
            'api_connected': True
        }
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration (without sensitive data)"""
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()

        # Return configuration without sensitive data
        safe_config = {
            'deepseek_api_base': config.deepseek_api_base,
            'deepseek_model': config.deepseek_model,
            'deepseek_max_tokens': config.deepseek_max_tokens,
            'deepseek_temperature': config.deepseek_temperature,
            'deepseek_timeout': config.deepseek_timeout,
            'log_level': config.log_level,
            'cache_enabled': config.cache_enabled,
            'cache_ttl_seconds': config.cache_ttl_seconds,
            'max_concurrent_requests': config.max_concurrent_requests,
            'enable_input_validation': config.enable_input_validation,
            'enable_output_sanitization': config.enable_output_sanitization,
            'max_input_length': config.max_input_length,
            'sensitive_data_masking': config.sensitive_data_masking,
            'enable_performance_monitoring': config.enable_performance_monitoring
        }

        return jsonify({
            'success': True,
            'data': safe_config
        })

    except Exception as e:
        logging.error(f"Error retrieving configuration: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration (admin endpoint)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No configuration data provided'
            }), 400

        config_manager = ConfigManager()
        config_manager.update_config(data)

        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })

    except ConfigurationError as e:
        return jsonify({
            'success': False,
            'error': f'Configuration validation failed: {e}'
        }), 400
    except Exception as e:
        logging.error(f"Error updating configuration: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance statistics"""
    try:
        stats = get_performance_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logging.error(f"Error retrieving performance stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/performance/cleanup', methods=['POST'])
def trigger_memory_cleanup():
    """Trigger memory cleanup"""
    try:
        memory_saved = cleanup_memory()
        return jsonify({
            'success': True,
            'memory_saved_mb': memory_saved
        })
    except Exception as e:
        logging.error(f"Error during memory cleanup: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'COTCAgent API',
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("Starting COTCAgent API server...")
    print("Access URL: http://localhost:5000")
    print("API documentation: http://localhost:5000/api/health")

    # Create templates directory
    os.makedirs('templates', exist_ok=True)

    # Move HTML file to templates directory
    if os.path.exists('web_interface.html'):
        import shutil
        shutil.move('web_interface.html', 'templates/web_interface.html')

    app.run(debug=True, host='0.0.0.0', port=5000)
