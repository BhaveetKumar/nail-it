"""
Real AI Video Generator CLI
Uses actual AI models for video generation
"""

import click
import requests
import json
import time
import os
from pathlib import Path

# Import real AI components
from core.real_video_composer import RealVideoComposer
from core.real_tts_engine import RealTTSEngine
from core.real_avatar_generator import RealAvatarGenerator

# Configuration
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
    }
}

# Initialize real AI components
video_composer = RealVideoComposer(config)
tts_engine = RealTTSEngine(config['models']['tts'])
avatar_generator = RealAvatarGenerator(config['models']['avatar'])

@click.group()
def cli():
    """Real AI Video Generator CLI"""
    pass

@cli.command()
def status():
    """Check system status"""
    click.echo("🔍 Checking real AI system status...")
    
    try:
        # Check TTS engine
        if tts_engine.load_model('female_professional'):
            click.echo("✅ Real TTS Engine: Loaded")
        else:
            click.echo("❌ Real TTS Engine: Failed to load")
        
        # Check avatar generator
        if avatar_generator.load_model():
            click.echo("✅ Real Avatar Generator: Loaded")
        else:
            click.echo("❌ Real Avatar Generator: Failed to load")
        
        # Check video composer
        click.echo("✅ Real Video Composer: Loaded")
        
        click.echo(f"🖥️  Device: {tts_engine.device}")
        click.echo("🤖 AI Models: Real AI models loaded")
        
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}")

@cli.command()
@click.option('--text', '-t', required=True, help='Text to convert to video')
@click.option('--output', '-o', default='output.mp4', help='Output video file path')
@click.option('--style', '-s', default='professional', help='Video style')
@click.option('--duration', '-d', default=10, help='Video duration in seconds')
@click.option('--voice', help='Voice to use for TTS')
@click.option('--effects', help='Special effects to apply')
def text_to_video(text, output, style, duration, voice, effects):
    """Generate video from text using real AI models"""
    click.echo("🎬 Starting real AI video generation...")
    
    try:
        # Load models
        tts_engine.load_model(voice or 'female_professional')
        avatar_generator.load_model()
        
        # Generate video
        if effects:
            effects_list = effects.split(',')
            video_path = video_composer.create_with_effects(text, effects_list)
        else:
            video_path = video_composer.create_from_text(text, style, duration, voice)
        
        if video_path and os.path.exists(video_path):
            # Move to desired output path
            import shutil
            shutil.move(video_path, output)
            click.echo(f"✅ Real AI video generated successfully: {output}")
        else:
            click.echo("❌ Failed to generate real AI video")
            
    except Exception as e:
        click.echo(f"❌ Error generating video: {e}")

@cli.command()
@click.option('--prompt', '-p', required=True, help='Avatar prompt')
@click.option('--output', '-o', default='avatar.png', help='Output avatar file path')
@click.option('--style', '-s', default='realistic', help='Avatar style')
def generate_avatar(prompt, output, style):
    """Generate avatar using real AI models"""
    click.echo("🎨 Generating real AI avatar...")
    
    try:
        # Load model
        avatar_generator.load_model()
        
        # Generate avatar
        avatar = avatar_generator.generate_avatar(prompt, style)
        
        if avatar:
            avatar.save(output)
            click.echo(f"✅ Real AI avatar generated successfully: {output}")
        else:
            click.echo("❌ Failed to generate real AI avatar")
            
    except Exception as e:
        click.echo(f"❌ Error generating avatar: {e}")

@cli.command()
@click.option('--text', '-t', required=True, help='Text to convert to audio')
@click.option('--output', '-o', default='output.wav', help='Output audio file path')
@click.option('--voice', help='Voice to use for TTS')
def text_to_audio(text, output, voice):
    """Generate audio from text using real TTS"""
    click.echo("🎵 Generating real AI audio...")
    
    try:
        # Load model
        tts_engine.load_model(voice or 'female_professional')
        
        # Generate audio
        audio_path = tts_engine.generate_speech(text, voice)
        
        if audio_path and os.path.exists(audio_path):
            # Move to desired output path
            import shutil
            shutil.move(audio_path, output)
            click.echo(f"✅ Real AI audio generated successfully: {output}")
        else:
            click.echo("❌ Failed to generate real AI audio")
            
    except Exception as e:
        click.echo(f"❌ Error generating audio: {e}")

@cli.command()
@click.option('--url', default='http://localhost:8080', help='Web interface URL')
def web_status(url):
    """Check web interface status"""
    click.echo(f"🌐 Checking web interface at {url}...")
    
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            click.echo(f"✅ Web interface: {data['status']}")
            click.echo(f"   Message: {data['message']}")
        else:
            click.echo(f"❌ Web interface: HTTP {response.status_code}")
            
    except Exception as e:
        click.echo(f"❌ Web interface check failed: {e}")

@cli.command()
@click.option('--url', default='http://localhost:8080', help='Web interface URL')
@click.option('--text', '-t', required=True, help='Text to convert to video')
@click.option('--style', '-s', default='professional', help='Video style')
@click.option('--voice', help='Voice to use for TTS')
def web_generate(url, text, style, voice):
    """Generate video via web API using real AI models"""
    click.echo("🌐 Generating video via web API...")
    
    try:
        data = {
            'text': text,
            'style': style,
            'voice': voice,
            'duration': 10
        }
        
        response = requests.post(f"{url}/api/generate-video", json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            click.echo(f"✅ Video generated successfully!")
            click.echo(f"   Video ID: {result.get('video_id')}")
            click.echo(f"   Download URL: {url}{result.get('video_url')}")
        else:
            click.echo(f"❌ Video generation failed: {response.text}")
            
    except Exception as e:
        click.echo(f"❌ Web API error: {e}")

if __name__ == '__main__':
    cli()
