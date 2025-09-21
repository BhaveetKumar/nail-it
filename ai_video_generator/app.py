"""
Web Interface for AI Video Generator
Flask-based web application for easy video generation
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

# Import core components
from core.video_composer import VideoComposer, AdvancedVideoComposer
from core.tts_engine import TTSEngine
from core.avatar_generator import AvatarGenerator
from core.talking_head import TalkingHead
from core.scene_generator import SceneGenerator

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
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
video_composer = AdvancedVideoComposer(config)
tts_engine = TTSEngine(config['models']['tts'])
avatar_generator = AvatarGenerator(config['models']['avatar'])
talking_head = TalkingHead(config['models']['talking_head'])
scene_generator = SceneGenerator(config['models']['video'])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_video():
    """Generate video from text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        style = data.get('style', 'professional')
        duration = data.get('duration', 10)
        voice = data.get('voice', None)
        effects = data.get('effects', [])
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate video
        if effects:
            video_path = video_composer.create_with_effects(text, effects)
        else:
            video_path = video_composer.create_from_text(text, style, duration, voice)
        
        if not video_path:
            return jsonify({'error': 'Failed to generate video'}), 500
        
        # Move to output folder
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        os.rename(video_path, output_path)
        
        return jsonify({
            'success': True,
            'video_url': f'/download/{output_filename}',
            'message': 'Video generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-from-script', methods=['POST'])
def generate_from_script():
    """Generate video from uploaded script"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Generate video from script
            video_path = video_composer.create_from_script(file_path)
            
            if not video_path:
                return jsonify({'error': 'Failed to generate video from script'}), 500
            
            # Move to output folder
            output_filename = f"{uuid.uuid4()}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            os.rename(video_path, output_path)
            
            # Cleanup uploaded file
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'video_url': f'/download/{output_filename}',
                'message': 'Video generated from script successfully'
            })
        
    except Exception as e:
        logger.error(f"Error generating video from script: {e}")
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

@app.route('/api/avatar', methods=['POST'])
def generate_avatar():
    """Generate avatar image"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'professional person')
        style = data.get('style', 'realistic')
        
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
            'message': 'Avatar generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating avatar: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview-audio', methods=['POST'])
def preview_audio():
    """Generate audio preview"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', None)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate audio
        audio_path = tts_engine.generate_speech(text, voice)
        if not audio_path:
            return jsonify({'error': 'Failed to generate audio'}), 500
        
        # Move to output folder
        audio_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], audio_filename)
        os.rename(audio_path, output_path)
        
        return jsonify({
            'success': True,
            'audio_url': f'/download/{audio_filename}',
            'message': 'Audio preview generated successfully'
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

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        status = {
            'tts_engine': 'loaded' if tts_engine else 'not loaded',
            'avatar_generator': 'loaded' if avatar_generator else 'not loaded',
            'talking_head': 'loaded' if talking_head else 'not loaded',
            'scene_generator': 'loaded' if scene_generator else 'not loaded',
            'video_composer': 'loaded' if video_composer else 'not loaded'
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get available templates"""
    try:
        templates = config.get('templates', {})
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
    # Load models on startup
    logger.info("Loading AI models...")
    
    try:
        tts_engine.load_model('female_professional')
        logger.info("TTS engine loaded")
    except Exception as e:
        logger.error(f"Failed to load TTS engine: {e}")
    
    try:
        avatar_generator.load_model()
        logger.info("Avatar generator loaded")
    except Exception as e:
        logger.error(f"Failed to load avatar generator: {e}")
    
    try:
        talking_head.load_model()
        logger.info("Talking head loaded")
    except Exception as e:
        logger.error(f"Failed to load talking head: {e}")
    
    try:
        scene_generator.load_model()
        logger.info("Scene generator loaded")
    except Exception as e:
        logger.error(f"Failed to load scene generator: {e}")
    
    logger.info("Starting AI Video Generator web interface...")
    app.run(
        host=config['web']['host'],
        port=config['web']['port'],
        debug=config['web']['debug']
    )
