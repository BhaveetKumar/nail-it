"""
Text-to-Speech Engine for AI Video Generator
Supports multiple TTS models and voice options
"""

import torch
import torchaudio
from TTS.api import TTS
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class TTSEngine:
    """Text-to-Speech engine with multiple model support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._get_device()
        self.models = {}
        self.current_voice = None
        
    def _get_device(self) -> str:
        """Determine the best device for TTS processing"""
        if torch.cuda.is_available() and self.config.get('device') != 'cpu':
            return 'cuda'
        return 'cpu'
    
    def load_model(self, voice_name: str) -> bool:
        """Load a specific TTS model"""
        try:
            voice_config = self.config['voices'][voice_name]
            model_name = voice_config['model']
            
            logger.info(f"Loading TTS model: {model_name}")
            tts = TTS(model_name).to(self.device)
            
            self.models[voice_name] = {
                'model': tts,
                'config': voice_config
            }
            
            logger.info(f"Successfully loaded voice: {voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load voice {voice_name}: {e}")
            return False
    
    def generate_speech(self, text: str, voice_name: str = None, 
                       output_path: str = None) -> Optional[str]:
        """Generate speech from text"""
        try:
            if voice_name is None:
                voice_name = self.config.get('default_voice', 'female_professional')
            
            if voice_name not in self.models:
                if not self.load_model(voice_name):
                    return None
            
            model_info = self.models[voice_name]
            tts = model_info['model']
            config = model_info['config']
            
            # Generate audio
            if output_path is None:
                output_path = f"temp_audio_{voice_name}.wav"
            
            # Handle different model types
            if 'speaker' in config:
                tts.tts_to_file(text=text, file_path=output_path, 
                              speaker=config['speaker'])
            else:
                tts.tts_to_file(text=text, file_path=output_path)
            
            logger.info(f"Generated speech: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            return None
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        return list(self.config['voices'].keys())
    
    def get_voice_info(self, voice_name: str) -> Optional[Dict]:
        """Get information about a specific voice"""
        if voice_name in self.config['voices']:
            return self.config['voices'][voice_name]
        return None
    
    def generate_batch(self, texts: List[str], voice_name: str = None) -> List[str]:
        """Generate speech for multiple texts"""
        results = []
        for i, text in enumerate(texts):
            output_path = f"batch_audio_{i}_{voice_name or 'default'}.wav"
            result = self.generate_speech(text, voice_name, output_path)
            results.append(result)
        return results
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary audio files"""
        for file_path in file_paths:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")

class AdvancedTTSEngine(TTSEngine):
    """Advanced TTS engine with emotion and style control"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.emotion_models = {}
        self.style_models = {}
    
    def generate_with_emotion(self, text: str, voice_name: str, 
                            emotion: str = "neutral") -> Optional[str]:
        """Generate speech with specific emotion"""
        # This would require emotion-aware TTS models
        # For now, we'll use the base implementation
        return self.generate_speech(text, voice_name)
    
    def generate_with_style(self, text: str, voice_name: str, 
                           style: str = "normal") -> Optional[str]:
        """Generate speech with specific style (fast, slow, dramatic, etc.)"""
        # This would require style-aware TTS models
        # For now, we'll use the base implementation
        return self.generate_speech(text, voice_name)
    
    def adjust_speed(self, audio_path: str, speed_factor: float) -> str:
        """Adjust the speed of generated audio"""
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Adjust speed
            y_fast = librosa.effects.time_stretch(y, rate=speed_factor)
            
            # Save adjusted audio
            output_path = audio_path.replace('.wav', f'_speed_{speed_factor}.wav')
            sf.write(output_path, y_fast, sr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to adjust speed: {e}")
            return audio_path
    
    def add_silence(self, audio_path: str, silence_duration: float = 0.5) -> str:
        """Add silence to the beginning and end of audio"""
        try:
            y, sr = sf.read(audio_path)
            silence = np.zeros(int(silence_duration * sr))
            
            # Add silence at beginning and end
            y_with_silence = np.concatenate([silence, y, silence])
            
            output_path = audio_path.replace('.wav', '_with_silence.wav')
            sf.write(output_path, y_with_silence, sr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to add silence: {e}")
            return audio_path
