"""
Real AI Video Generator Web Interface
Uses actual AI models for video generation
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import os
import yaml
import logging
from pathlib import Path
import tempfile
import uuid
from werkzeug.utils import secure_filename

# Import real AI components
from core.real_video_composer import RealVideoComposer
from core.real_tts_engine import RealTTSEngine
from core.real_avatar_generator import RealAvatarGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Create default config if file doesn't exist
    config = {
        'models': {
            'tts': {
                'default_voice': 'female_professional',
                'voices': {
                    'female_professional': {'model': 'microsoft/speecht5_tts', 'speaker': 'default'},
                    'male_deep': {'model': 'microsoft/speecht5_tts', 'speaker': 'p225'},
                    'female_young': {'model': 'microsoft/speecht5_tts', 'speaker': 'p226'}
                }
            },
            'avatar': {
                'default': 'runwayml/stable-diffusion-v1-5',
                'style': 'realistic',
                'resolution': [512, 512]
            },
            'video': {
                'resolution': [1280, 720],
                'fps': 24
            }
        },
        'web': {
            'host': '0.0.0.0',
            'port': 8080,
            'debug': True
        }
    }

# Initialize real AI components
video_composer = RealVideoComposer(config)
tts_engine = RealTTSEngine(config['models']['tts'])
avatar_generator = RealAvatarGenerator(config['models']['avatar'])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    """Generate video from text using real AI models"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        style = data.get('style', 'professional')
        duration = data.get('duration', 10)
        voice = data.get('voice', None)
        effects = data.get('effects', [])
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        logger.info(f"Generating real video: '{text[:50]}...' with style '{style}'")
        
        # Generate video using real AI models
        if effects:
            video_path = video_composer.create_with_effects(text, effects)
        else:
            video_path = video_composer.create_from_text(text, style, duration, voice)
        
        if not video_path:
            return jsonify({'error': 'Failed to generate video'}), 500
        
        # Move to output folder
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Copy the video file to output folder
        import shutil
        shutil.copy2(video_path, output_path)
        
        # Clean up original file
        if os.path.exists(video_path):
            os.unlink(video_path)
        
        return jsonify({
            'success': True,
            'video_id': output_filename,
            'status': 'completed',
            'video_url': f'/download/{output_filename}',
            'message': 'Real AI video generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/avatar', methods=['POST'])
def generate_avatar():
    """Generate avatar image using real AI models"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'professional person')
        style = data.get('style', 'realistic')
        
        logger.info(f"Generating real avatar: '{prompt}' with style '{style}'")
        
        avatar = avatar_generator.generate_avatar(prompt, style)
        if not avatar:
            return jsonify({'error': 'Failed to generate avatar'}), 500
        
        # Save avatar
        avatar_filename = f"{uuid.uuid4()}.png"
        avatar_path = os.path.join(app.config['OUTPUT_FOLDER'], avatar_filename)
        avatar.save(avatar_path)
        
        return jsonify({
            'success': True,
            'avatar_url': f'/download/{avatar_filename}',
            'message': 'Real AI avatar generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating avatar: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview-audio', methods=['POST'])
def preview_audio():
    """Generate audio preview using real TTS"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', None)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        logger.info(f"Generating real audio preview: '{text[:50]}...'")
        
        # Generate audio using real TTS
        audio_path = tts_engine.generate_speech(text, voice)
        if not audio_path:
            return jsonify({'error': 'Failed to generate audio'}), 500
        
        # Move to output folder
        audio_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], audio_filename)
        
        # Copy the audio file to output folder
        import shutil
        shutil.copy2(audio_path, output_path)
        
        # Clean up original file
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        
        return jsonify({
            'success': True,
            'audio_url': f'/download/{audio_filename}',
            'message': 'Real AI audio preview generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating audio preview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated file"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Real AI Video Generator is running'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        status = {
            'tts_engine': 'loaded' if tts_engine else 'not loaded',
            'avatar_generator': 'loaded' if avatar_generator else 'not loaded',
            'video_composer': 'loaded' if video_composer else 'not loaded',
            'ai_models': 'real_ai_models',
            'device': 'cuda' if tts_engine.device == 'cuda' else 'cpu'
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Get available voices"""
    try:
        voices = tts_engine.get_available_voices()
        return jsonify({'voices': voices})
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get available templates"""
    try:
        templates = {
            'professional': {
                'background': '#ffffff',
                'text_color': '#333333',
                'font_family': 'Arial',
                'font_size': 24,
                'layout': 'center'
            },
            'modern': {
                'background': '#1a1a1a',
                'text_color': '#ffffff',
                'font_family': 'Helvetica',
                'font_size': 28,
                'layout': 'left'
            },
            'creative': {
                'background': '#667eea',
                'text_color': '#ffffff',
                'font_family': 'Georgia',
                'font_size': 26,
                'layout': 'dynamic'
            }
        }
        return jsonify({'templates': templates})
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 error"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 error"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load real AI models on startup
    logger.info("Loading real AI models...")
    
    try:
        tts_engine.load_model('female_professional')
        logger.info("Real TTS engine loaded")
    except Exception as e:
        logger.error(f"Failed to load TTS engine: {e}")
    
    try:
        avatar_generator.load_model()
        logger.info("Real avatar generator loaded")
    except Exception as e:
        logger.error(f"Failed to load avatar generator: {e}")
    
    logger.info("Starting Real AI Video Generator web interface...")
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )
