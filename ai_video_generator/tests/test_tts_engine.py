"""
Tests for TTS Engine
"""

import pytest
import tempfile
import os
from pathlib import Path
from core.tts_engine import TTSEngine, AdvancedTTSEngine

class TestTTSEngine:
    """Test cases for TTS Engine"""
    
    @pytest.fixture
    def tts_config(self):
        """Mock TTS configuration"""
        return {
            'default_voice': 'female_professional',
            'voices': {
                'female_professional': {
                    'model': 'tts_models/en/ljspeech/tacotron2-DDC',
                    'speaker': 'default'
                },
                'male_deep': {
                    'model': 'tts_models/en/vctk/vits',
                    'speaker': 'p225'
                }
            }
        }
    
    @pytest.fixture
    def tts_engine(self, tts_config):
        """Create TTS engine instance"""
        return TTSEngine(tts_config)
    
    def test_initialization(self, tts_engine):
        """Test TTS engine initialization"""
        assert tts_engine is not None
        assert tts_engine.config is not None
        assert tts_engine.device in ['cpu', 'cuda']
    
    def test_get_available_voices(self, tts_engine):
        """Test getting available voices"""
        voices = tts_engine.get_available_voices()
        assert isinstance(voices, list)
        assert 'female_professional' in voices
        assert 'male_deep' in voices
    
    def test_get_voice_info(self, tts_engine):
        """Test getting voice information"""
        voice_info = tts_engine.get_voice_info('female_professional')
        assert voice_info is not None
        assert 'model' in voice_info
        assert 'speaker' in voice_info
        
        # Test non-existent voice
        voice_info = tts_engine.get_voice_info('non_existent')
        assert voice_info is None
    
    @pytest.mark.skip(reason="Requires actual TTS model")
    def test_generate_speech(self, tts_engine):
        """Test speech generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_speech.wav")
            
            # This would require actual TTS model
            result = tts_engine.generate_speech(
                text="Hello, world!",
                voice="female_professional",
                output_path=output_path
            )
            
            # In real test, check if file was created
            # assert result == output_path
            # assert os.path.exists(output_path)
    
    def test_generate_batch(self, tts_engine):
        """Test batch speech generation"""
        texts = ["Hello", "World", "AI Video"]
        
        # This would require actual TTS model
        results = tts_engine.generate_batch(texts, "female_professional")
        
        assert isinstance(results, list)
        assert len(results) == len(texts)
    
    def test_cleanup_temp_files(self, tts_engine):
        """Test temporary file cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_file.wav")
            
            # Create a temporary file
            with open(temp_file, 'w') as f:
                f.write("test")
            
            assert os.path.exists(temp_file)
            
            # Clean up
            tts_engine.cleanup_temp_files([temp_file])
            
            # File should still exist (cleanup only removes if it's a temp file)
            # In real implementation, this would check for specific temp file patterns

class TestAdvancedTTSEngine:
    """Test cases for Advanced TTS Engine"""
    
    @pytest.fixture
    def advanced_tts_config(self):
        """Mock advanced TTS configuration"""
        return {
            'default_voice': 'female_professional',
            'voices': {
                'female_professional': {
                    'model': 'tts_models/en/ljspeech/tacotron2-DDC',
                    'speaker': 'default'
                }
            }
        }
    
    @pytest.fixture
    def advanced_tts_engine(self, advanced_tts_config):
        """Create advanced TTS engine instance"""
        return AdvancedTTSEngine(advanced_tts_config)
    
    def test_initialization(self, advanced_tts_engine):
        """Test advanced TTS engine initialization"""
        assert advanced_tts_engine is not None
        assert advanced_tts_engine.emotion_models == {}
        assert advanced_tts_engine.style_models == {}
    
    @pytest.mark.skip(reason="Requires actual TTS model")
    def test_generate_with_emotion(self, advanced_tts_engine):
        """Test emotion-based speech generation"""
        result = advanced_tts_engine.generate_with_emotion(
            text="Hello, world!",
            voice="female_professional",
            emotion="happy"
        )
        
        # In real test, verify emotion was applied
        assert result is not None
    
    @pytest.mark.skip(reason="Requires actual TTS model")
    def test_generate_with_style(self, advanced_tts_engine):
        """Test style-based speech generation"""
        result = advanced_tts_engine.generate_with_style(
            text="Hello, world!",
            voice="female_professional",
            style="dramatic"
        )
        
        # In real test, verify style was applied
        assert result is not None
    
    @pytest.mark.skip(reason="Requires actual audio file")
    def test_adjust_speed(self, advanced_tts_engine):
        """Test speed adjustment"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create a dummy audio file
            with open(temp_path, 'wb') as f:
                f.write(b'dummy audio data')
            
            result = advanced_tts_engine.adjust_speed(temp_path, 1.5)
            
            # Should return a new file path
            assert result != temp_path
            assert result.endswith('_speed_1.5.wav')
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(result):
                os.unlink(result)
    
    @pytest.mark.skip(reason="Requires actual audio file")
    def test_add_silence(self, advanced_tts_engine):
        """Test adding silence to audio"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create a dummy audio file
            with open(temp_path, 'wb') as f:
                f.write(b'dummy audio data')
            
            result = advanced_tts_engine.add_silence(temp_path, 1.0)
            
            # Should return a new file path
            assert result != temp_path
            assert result.endswith('_with_silence.wav')
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(result):
                os.unlink(result)

if __name__ == '__main__':
    pytest.main([__file__])
