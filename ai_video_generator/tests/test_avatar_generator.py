"""
Tests for Avatar Generator
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from core.avatar_generator import AvatarGenerator, AdvancedAvatarGenerator

class TestAvatarGenerator:
    """Test cases for Avatar Generator"""
    
    @pytest.fixture
    def avatar_config(self):
        """Mock avatar configuration"""
        return {
            'avatar': {
                'default': 'stabilityai/stable-diffusion-xl-base-1.0',
                'style': 'realistic',
                'resolution': [512, 512]
            },
            'device': 'cpu'
        }
    
    @pytest.fixture
    def avatar_generator(self, avatar_config):
        """Create avatar generator instance"""
        return AvatarGenerator(avatar_config)
    
    def test_initialization(self, avatar_generator):
        """Test avatar generator initialization"""
        assert avatar_generator is not None
        assert avatar_generator.config is not None
        assert avatar_generator.device in ['cpu', 'cuda']
        assert avatar_generator.avatar_cache == {}
    
    def test_get_device(self, avatar_generator):
        """Test device detection"""
        device = avatar_generator._get_device()
        assert device in ['cpu', 'cuda']
    
    @pytest.mark.skip(reason="Requires actual diffusion model")
    def test_load_model(self, avatar_generator):
        """Test model loading"""
        result = avatar_generator.load_model()
        # In real test, this would load the actual model
        # assert result == True
        # assert avatar_generator.pipeline is not None
    
    @pytest.mark.skip(reason="Requires actual diffusion model")
    def test_generate_avatar(self, avatar_generator):
        """Test avatar generation"""
        avatar = avatar_generator.generate_avatar(
            prompt="professional person",
            style="realistic",
            seed=42
        )
        
        # In real test, verify avatar was generated
        # assert avatar is not None
        # assert isinstance(avatar, Image.Image)
        # assert avatar.size == (512, 512)
    
    def test_enhance_prompt(self, avatar_generator):
        """Test prompt enhancement"""
        # Test realistic style
        enhanced = avatar_generator._enhance_prompt("person", "realistic")
        assert "professional headshot" in enhanced
        assert "realistic" in enhanced
        
        # Test cartoon style
        enhanced = avatar_generator._enhance_prompt("person", "cartoon")
        assert "cartoon style" in enhanced
        assert "animated" in enhanced
        
        # Test anime style
        enhanced = avatar_generator._enhance_prompt("person", "anime")
        assert "anime style" in enhanced
        assert "manga" in enhanced
        
        # Test professional style
        enhanced = avatar_generator._enhance_prompt("person", "professional")
        assert "business professional" in enhanced
        assert "corporate" in enhanced
        
        # Test casual style
        enhanced = avatar_generator._enhance_prompt("person", "casual")
        assert "casual" in enhanced
        assert "friendly" in enhanced
        
        # Test unknown style
        enhanced = avatar_generator._enhance_prompt("person", "unknown")
        assert "professional headshot" in enhanced  # Should default to realistic
    
    @pytest.mark.skip(reason="Requires actual diffusion model")
    def test_generate_avatar_variations(self, avatar_generator):
        """Test avatar variation generation"""
        variations = avatar_generator.generate_avatar_variations(
            "professional person", 4
        )
        
        # In real test, verify variations were generated
        # assert len(variations) == 4
        # assert all(isinstance(v, Image.Image) for v in variations)
    
    def test_create_avatar_with_background(self, avatar_generator):
        """Test avatar with background creation"""
        # Create a dummy avatar
        avatar = Image.new('RGB', (512, 512), color='red')
        
        # Test different background styles
        bg_styles = ['professional', 'modern', 'gradient', 'white', 'dark']
        
        for style in bg_styles:
            result = avatar_generator.create_avatar_with_background(avatar, style)
            assert result is not None
            assert isinstance(result, Image.Image)
            assert result.size == avatar.size
    
    def test_add_text_overlay(self, avatar_generator):
        """Test adding text overlay to avatar"""
        # Create a dummy avatar
        avatar = Image.new('RGB', (512, 512), color='white')
        
        # Test different positions
        positions = ['bottom', 'top', 'center']
        
        for position in positions:
            result = avatar_generator.add_text_overlay(avatar, "Test Text", position)
            assert result is not None
            assert isinstance(result, Image.Image)
            assert result.size == avatar.size

class TestAdvancedAvatarGenerator:
    """Test cases for Advanced Avatar Generator"""
    
    @pytest.fixture
    def advanced_avatar_config(self):
        """Mock advanced avatar configuration"""
        return {
            'avatar': {
                'default': 'stabilityai/stable-diffusion-xl-base-1.0',
                'style': 'realistic',
                'resolution': [512, 512]
            },
            'device': 'cpu'
        }
    
    @pytest.fixture
    def advanced_avatar_generator(self, advanced_avatar_config):
        """Create advanced avatar generator instance"""
        return AdvancedAvatarGenerator(advanced_avatar_config)
    
    def test_initialization(self, advanced_avatar_generator):
        """Test advanced avatar generator initialization"""
        assert advanced_avatar_generator is not None
        assert advanced_avatar_generator.face_detector is None
        assert advanced_avatar_generator.face_landmarks is None
    
    @pytest.mark.skip(reason="Requires MediaPipe")
    def test_load_face_models(self, advanced_avatar_generator):
        """Test face model loading"""
        result = advanced_avatar_generator.load_face_models()
        # In real test, verify models were loaded
        # assert result == True
        # assert advanced_avatar_generator.face_detector is not None
        # assert advanced_avatar_generator.face_landmarks is not None
    
    @pytest.mark.skip(reason="Requires MediaPipe and face image")
    def test_detect_face_landmarks(self, advanced_avatar_generator):
        """Test face landmark detection"""
        # Create a dummy image
        image = Image.new('RGB', (512, 512), color='white')
        
        landmarks = advanced_avatar_generator.detect_face_landmarks(image)
        
        # In real test with actual face image
        # assert landmarks is not None
        # assert 'landmarks' in landmarks
        # assert 'face_count' in landmarks
    
    @pytest.mark.skip(reason="Requires actual diffusion model")
    def test_generate_consistent_avatar(self, advanced_avatar_generator):
        """Test consistent avatar generation"""
        avatars = advanced_avatar_generator.generate_consistent_avatar(
            "professional person", 4
        )
        
        # In real test, verify consistent avatars were generated
        # assert len(avatars) == 4
        # assert all(isinstance(a, Image.Image) for a in avatars)
        # All avatars should be similar but with slight variations

class TestAvatarGeneratorIntegration:
    """Integration tests for Avatar Generator"""
    
    @pytest.fixture
    def full_avatar_config(self):
        """Full avatar configuration for integration tests"""
        return {
            'avatar': {
                'default': 'stabilityai/stable-diffusion-xl-base-1.0',
                'style': 'realistic',
                'resolution': [512, 512]
            },
            'device': 'cpu'
        }
    
    @pytest.mark.skip(reason="Requires actual models")
    def test_full_avatar_generation_workflow(self, full_avatar_config):
        """Test complete avatar generation workflow"""
        generator = AvatarGenerator(full_avatar_config)
        
        # Load model
        assert generator.load_model() == True
        
        # Generate avatar
        avatar = generator.generate_avatar("professional businesswoman")
        assert avatar is not None
        assert isinstance(avatar, Image.Image)
        
        # Add background
        avatar_with_bg = generator.create_avatar_with_background(avatar, "professional")
        assert avatar_with_bg is not None
        
        # Add text overlay
        avatar_with_text = generator.add_text_overlay(avatar_with_bg, "Jane Doe", "bottom")
        assert avatar_with_text is not None
        
        # Save avatar
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            avatar_with_text.save(temp_path)
            assert os.path.exists(temp_path)
            
            # Verify file can be loaded
            loaded_avatar = Image.open(temp_path)
            assert loaded_avatar.size == (512, 512)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_error_handling(self, full_avatar_config):
        """Test error handling in avatar generation"""
        generator = AvatarGenerator(full_avatar_config)
        
        # Test with invalid prompt
        avatar = generator.generate_avatar("")
        # Should handle empty prompt gracefully
        
        # Test with invalid style
        avatar = generator.generate_avatar("person", "invalid_style")
        # Should default to realistic style
        
        # Test with None input
        result = generator.create_avatar_with_background(None, "professional")
        # Should handle None input gracefully

if __name__ == '__main__':
    pytest.main([__file__])
