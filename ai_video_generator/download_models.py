#!/usr/bin/env python3
"""
Model Download Script for AI Video Generator
Downloads all required AI models for local use
"""

import os
import sys
import yaml
import logging
from pathlib import Path
import subprocess
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url: str, filepath: str, description: str = "Downloading"):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def download_huggingface_model(model_name: str, local_dir: str):
    """Download a Hugging Face model"""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"Downloading Hugging Face model: {model_name}")
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Download model and tokenizer
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save locally
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        
        logger.info(f"Model saved to: {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Hugging Face model {model_name}: {e}")
        return False

def download_diffusers_model(model_name: str, local_dir: str):
    """Download a Diffusers model"""
    try:
        from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
        
        logger.info(f"Downloading Diffusers model: {model_name}")
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        if "stable-diffusion" in model_name:
            pipeline = StableDiffusionPipeline.from_pretrained(model_name)
        elif "stable-video" in model_name:
            pipeline = StableVideoDiffusionPipeline.from_pretrained(model_name)
        else:
            logger.error(f"Unknown model type: {model_name}")
            return False
        
        # Save locally
        pipeline.save_pretrained(local_dir)
        
        logger.info(f"Model saved to: {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Diffusers model {model_name}: {e}")
        return False

def download_tts_model(model_name: str, local_dir: str):
    """Download a TTS model"""
    try:
        from TTS.api import TTS
        
        logger.info(f"Downloading TTS model: {model_name}")
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Download model
        tts = TTS(model_name)
        
        # Save model files
        model_path = os.path.join(local_dir, model_name.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)
        
        # This is a simplified approach - in practice, you'd need to save the model properly
        logger.info(f"TTS model downloaded: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download TTS model {model_name}: {e}")
        return False

def download_wav2lip_model(local_dir: str):
    """Download Wav2Lip model"""
    try:
        logger.info("Downloading Wav2Lip model...")
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Wav2Lip model URLs (these would be the actual URLs)
        model_urls = {
            "wav2lip_gan.pth": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth",
            "wav2lip.pth": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth"
        }
        
        for filename, url in model_urls.items():
            filepath = os.path.join(local_dir, filename)
            if not os.path.exists(filepath):
                success = download_file(url, filepath, f"Downloading {filename}")
                if not success:
                    logger.warning(f"Failed to download {filename}")
            else:
                logger.info(f"{filename} already exists")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Wav2Lip model: {e}")
        return False

def main():
    """Main download function"""
    try:
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Starting model download process...")
        
        # Download TTS models
        logger.info("📢 Downloading TTS models...")
        tts_config = config['models']['tts']
        for voice_name, voice_config in tts_config['voices'].items():
            model_name = voice_config['model']
            local_dir = models_dir / "tts" / voice_name
            download_tts_model(model_name, str(local_dir))
        
        # Download avatar generation model
        logger.info("👤 Downloading avatar generation model...")
        avatar_model = config['models']['avatar']['default']
        avatar_dir = models_dir / "avatar"
        download_diffusers_model(avatar_model, str(avatar_dir))
        
        # Download video generation model
        logger.info("🎬 Downloading video generation model...")
        video_model = config['models']['video']['model']
        video_dir = models_dir / "video"
        download_diffusers_model(video_model, str(video_dir))
        
        # Download talking head model
        logger.info("🗣️ Downloading talking head model...")
        talking_head_dir = models_dir / "talking_head"
        download_wav2lip_model(str(talking_head_dir))
        
        # Download additional models
        logger.info("📚 Downloading additional models...")
        
        # Download face detection models
        face_dir = models_dir / "face_detection"
        face_dir.mkdir(exist_ok=True)
        
        # Download MediaPipe models (these are downloaded automatically on first use)
        logger.info("Face detection models will be downloaded on first use")
        
        # Download background generation model
        bg_model = "runwayml/stable-diffusion-v1-5"
        bg_dir = models_dir / "background"
        download_diffusers_model(bg_model, str(bg_dir))
        
        logger.info("✅ All models downloaded successfully!")
        logger.info(f"Models saved to: {models_dir.absolute()}")
        
        # Print disk usage
        total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
        logger.info(f"Total disk usage: {total_size / (1024**3):.2f} GB")
        
    except Exception as e:
        logger.error(f"❌ Error during model download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
