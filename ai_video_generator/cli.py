#!/usr/bin/env python3
"""
Command Line Interface for AI Video Generator
"""

import click
import yaml
import logging
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.video_composer import VideoComposer, AdvancedVideoComposer
from core.tts_engine import TTSEngine
from core.avatar_generator import AvatarGenerator
from core.talking_head import TalkingHead
from core.scene_generator import SceneGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """AI Video Generator - Create videos from text using local AI models"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        with open(config, 'r') as f:
            ctx.obj['config'] = yaml.safe_load(f)
    except FileNotFoundError:
        click.echo(f"Error: Configuration file '{config}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        click.echo(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)

@cli.command()
@click.option('--text', '-t', required=True, help='Text to convert to video')
@click.option('--output', '-o', default='output.mp4', help='Output video file path')
@click.option('--style', '-s', default='professional', help='Video style (professional, modern, creative)')
@click.option('--duration', '-d', default=10, help='Video duration in seconds')
@click.option('--voice', help='Voice to use for TTS')
@click.option('--effects', multiple=True, help='Special effects to apply')
@click.pass_context
def text_to_video(ctx, text, output, style, duration, voice, effects):
    """Generate video from text"""
    try:
        click.echo("🎬 Starting video generation...")
        
        # Initialize video composer
        composer = AdvancedVideoComposer(ctx.obj['config'])
        
        # Generate video
        if effects:
            video_path = composer.create_with_effects(text, list(effects))
        else:
            video_path = composer.create_from_text(text, style, duration, voice)
        
        if video_path:
            # Move to desired output path
            Path(video_path).rename(output)
            click.echo(f"✅ Video generated successfully: {output}")
        else:
            click.echo("❌ Failed to generate video")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--script', '-s', required=True, help='Script file path')
@click.option('--output', '-o', default='output.mp4', help='Output video file path')
@click.pass_context
def script_to_video(ctx, script, output):
    """Generate video from script file"""
    try:
        click.echo("🎬 Starting script-to-video generation...")
        
        # Check if script file exists
        if not Path(script).exists():
            click.echo(f"❌ Script file not found: {script}")
            sys.exit(1)
        
        # Initialize video composer
        composer = VideoComposer(ctx.obj['config'])
        
        # Generate video
        video_path = composer.create_from_script(script)
        
        if video_path:
            # Move to desired output path
            Path(video_path).rename(output)
            click.echo(f"✅ Video generated successfully: {output}")
        else:
            click.echo("❌ Failed to generate video from script")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--prompt', '-p', required=True, help='Avatar description prompt')
@click.option('--output', '-o', default='avatar.png', help='Output image file path')
@click.option('--style', '-s', default='realistic', help='Avatar style')
@click.pass_context
def generate_avatar(ctx, prompt, output, style):
    """Generate avatar image"""
    try:
        click.echo("👤 Generating avatar...")
        
        # Initialize avatar generator
        generator = AvatarGenerator(ctx.obj['config']['models']['avatar'])
        
        # Generate avatar
        avatar = generator.generate_avatar(prompt, style)
        
        if avatar:
            avatar.save(output)
            click.echo(f"✅ Avatar generated successfully: {output}")
        else:
            click.echo("❌ Failed to generate avatar")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--text', '-t', required=True, help='Text to convert to speech')
@click.option('--output', '-o', default='speech.wav', help='Output audio file path')
@click.option('--voice', help='Voice to use')
@click.pass_context
def text_to_speech(ctx, text, output, voice):
    """Generate speech from text"""
    try:
        click.echo("🎤 Generating speech...")
        
        # Initialize TTS engine
        tts = TTSEngine(ctx.obj['config']['models']['tts'])
        
        # Generate speech
        audio_path = tts.generate_speech(text, voice)
        
        if audio_path:
            # Move to desired output path
            Path(audio_path).rename(output)
            click.echo(f"✅ Speech generated successfully: {output}")
        else:
            click.echo("❌ Failed to generate speech")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--avatar', '-a', required=True, help='Avatar image file path')
@click.option('--audio', required=True, help='Audio file path')
@click.option('--output', '-o', default='talking_head.mp4', help='Output video file path')
@click.pass_context
def talking_head(ctx, avatar, audio, output):
    """Create talking head video"""
    try:
        click.echo("🗣️ Creating talking head video...")
        
        # Check if files exist
        if not Path(avatar).exists():
            click.echo(f"❌ Avatar file not found: {avatar}")
            sys.exit(1)
        if not Path(audio).exists():
            click.echo(f"❌ Audio file not found: {audio}")
            sys.exit(1)
        
        # Initialize talking head
        talking_head = TalkingHead(ctx.obj['config']['models']['talking_head'])
        
        # Load avatar image
        from PIL import Image
        avatar_image = Image.open(avatar)
        
        # Create talking head
        video_path = talking_head.create_talking_head(avatar_image, audio)
        
        if video_path:
            # Move to desired output path
            Path(video_path).rename(output)
            click.echo(f"✅ Talking head video created: {output}")
        else:
            click.echo("❌ Failed to create talking head video")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.pass_context
def list_voices(ctx):
    """List available voices"""
    try:
        # Initialize TTS engine
        tts = TTSEngine(ctx.obj['config']['models']['tts'])
        
        # Get voices
        voices = tts.get_available_voices()
        
        if voices:
            click.echo("Available voices:")
            for voice in voices:
                click.echo(f"  - {voice}")
        else:
            click.echo("No voices available")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """Show system status"""
    try:
        click.echo("🔍 Checking system status...")
        
        # Check TTS engine
        try:
            tts = TTSEngine(ctx.obj['config']['models']['tts'])
            tts_status = "✅ Loaded" if tts.load_model('female_professional') else "❌ Failed"
        except:
            tts_status = "❌ Error"
        
        # Check avatar generator
        try:
            avatar_gen = AvatarGenerator(ctx.obj['config']['models']['avatar'])
            avatar_status = "✅ Loaded" if avatar_gen.load_model() else "❌ Failed"
        except:
            avatar_status = "❌ Error"
        
        # Check talking head
        try:
            talking_head = TalkingHead(ctx.obj['config']['models']['talking_head'])
            talking_head_status = "✅ Loaded" if talking_head.load_model() else "❌ Failed"
        except:
            talking_head_status = "❌ Error"
        
        # Check scene generator
        try:
            scene_gen = SceneGenerator(ctx.obj['config']['models']['video'])
            scene_status = "✅ Loaded" if scene_gen.load_model() else "❌ Failed"
        except:
            scene_status = "❌ Error"
        
        click.echo(f"TTS Engine: {tts_status}")
        click.echo(f"Avatar Generator: {avatar_status}")
        click.echo(f"Talking Head: {talking_head_status}")
        click.echo(f"Scene Generator: {scene_status}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--output', '-o', default='models', help='Output directory for models')
@click.pass_context
def download_models(ctx, output):
    """Download required AI models"""
    try:
        click.echo("📥 Downloading AI models...")
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(exist_ok=True)
        
        # Download TTS models
        click.echo("Downloading TTS models...")
        tts = TTSEngine(ctx.obj['config']['models']['tts'])
        for voice_name in tts.get_available_voices():
            click.echo(f"  Downloading {voice_name}...")
            tts.load_model(voice_name)
        
        # Download avatar generation model
        click.echo("Downloading avatar generation model...")
        avatar_gen = AvatarGenerator(ctx.obj['config']['models']['avatar'])
        avatar_gen.load_model()
        
        # Download video generation model
        click.echo("Downloading video generation model...")
        scene_gen = SceneGenerator(ctx.obj['config']['models']['video'])
        scene_gen.load_model()
        
        click.echo("✅ All models downloaded successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=5000, help='Port to bind to')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def web(ctx, host, port, debug):
    """Start web interface"""
    try:
        click.echo(f"🌐 Starting web interface on {host}:{port}")
        
        # Import and run the Flask app
        from app import app
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()
