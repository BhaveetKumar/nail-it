"""
AI Avatar Generator for AI Video Generator
Creates realistic avatars using local diffusion models
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AvatarGenerator:
    """AI Avatar generator using Stable Diffusion"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._get_device()
        self.pipeline = None
        self.avatar_cache = {}
        
    def _get_device(self) -> str:
        """Determine the best device for avatar generation"""
        if torch.cuda.is_available() and self.config.get('device') != 'cpu':
            return 'cuda'
        return 'cpu'
    
    def load_model(self) -> bool:
        """Load the Stable Diffusion model"""
        try:
            model_name = self.config['avatar']['default']
            logger.info(f"Loading avatar model: {model_name}")
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()
            
            logger.info("Avatar model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load avatar model: {e}")
            return False
    
    def generate_avatar(self, prompt: str, style: str = "realistic", 
                       seed: Optional[int] = None) -> Optional[Image.Image]:
        """Generate an avatar image"""
        try:
            if self.pipeline is None:
                if not self.load_model():
                    return None
            
            # Enhance prompt based on style
            enhanced_prompt = self._enhance_prompt(prompt, style)
            
            # Generate image
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)
            
            image = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt="blurry, low quality, distorted, deformed",
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            # Resize to target resolution
            target_size = tuple(self.config['avatar']['resolution'])
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            logger.info(f"Generated avatar: {prompt}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to generate avatar: {e}")
            return None
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt based on style"""
        style_enhancements = {
            "realistic": "professional headshot, high quality, detailed, realistic",
            "cartoon": "cartoon style, animated, colorful, friendly",
            "anime": "anime style, manga, detailed, colorful",
            "professional": "business professional, corporate, clean, modern",
            "casual": "casual, friendly, approachable, natural"
        }
        
        enhancement = style_enhancements.get(style, style_enhancements["realistic"])
        return f"{prompt}, {enhancement}"
    
    def generate_avatar_variations(self, base_prompt: str, count: int = 4) -> List[Image.Image]:
        """Generate multiple variations of an avatar"""
        variations = []
        for i in range(count):
            avatar = self.generate_avatar(base_prompt, seed=i * 1000)
            if avatar:
                variations.append(avatar)
        return variations
    
    def create_avatar_with_background(self, avatar: Image.Image, 
                                    background_style: str = "professional") -> Image.Image:
        """Add a background to the avatar"""
        try:
            # Create background
            bg_colors = {
                "professional": "#f8f9fa",
                "modern": "#1a1a1a",
                "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "white": "#ffffff",
                "dark": "#2c3e50"
            }
            
            bg_color = bg_colors.get(background_style, "#f8f9fa")
            
            # Create background image
            bg_image = Image.new('RGB', avatar.size, bg_color)
            
            # Composite avatar onto background
            if avatar.mode == 'RGBA':
                bg_image.paste(avatar, (0, 0), avatar)
            else:
                bg_image.paste(avatar, (0, 0))
            
            return bg_image
            
        except Exception as e:
            logger.error(f"Failed to create background: {e}")
            return avatar
    
    def add_text_overlay(self, image: Image.Image, text: str, 
                        position: str = "bottom") -> Image.Image:
        """Add text overlay to avatar image"""
        try:
            # Create a copy to avoid modifying original
            img_with_text = image.copy()
            draw = ImageDraw.Draw(img_with_text)
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            img_width, img_height = img_with_text.size
            
            if position == "bottom":
                x = (img_width - text_width) // 2
                y = img_height - text_height - 20
            elif position == "top":
                x = (img_width - text_width) // 2
                y = 20
            else:  # center
                x = (img_width - text_width) // 2
                y = (img_height - text_height) // 2
            
            # Draw text with outline
            draw.text((x-1, y-1), text, font=font, fill="black")
            draw.text((x+1, y-1), text, font=font, fill="black")
            draw.text((x-1, y+1), text, font=font, fill="black")
            draw.text((x+1, y+1), text, font=font, fill="black")
            draw.text((x, y), text, font=font, fill="white")
            
            return img_with_text
            
        except Exception as e:
            logger.error(f"Failed to add text overlay: {e}")
            return image

class AdvancedAvatarGenerator(AvatarGenerator):
    """Advanced avatar generator with more features"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.face_detector = None
        self.face_landmarks = None
    
    def load_face_models(self):
        """Load face detection and landmark models"""
        try:
            import mediapipe as mp
            
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            self.face_landmarks = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("Face models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load face models: {e}")
            return False
    
    def detect_face_landmarks(self, image: Image.Image) -> Optional[Dict]:
        """Detect face landmarks in the image"""
        try:
            if self.face_landmarks is None:
                if not self.load_face_models():
                    return None
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Process image
            results = self.face_landmarks.process(img_array)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                return {
                    'landmarks': landmarks,
                    'face_count': len(results.multi_face_landmarks)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect face landmarks: {e}")
            return None
    
    def generate_consistent_avatar(self, base_prompt: str, 
                                 variations: int = 4) -> List[Image.Image]:
        """Generate consistent avatars with slight variations"""
        avatars = []
        base_seed = 42  # Fixed base seed for consistency
        
        for i in range(variations):
            # Slight variations in prompt
            variation_prompts = [
                f"{base_prompt}, slight smile",
                f"{base_prompt}, looking slightly left",
                f"{base_prompt}, looking slightly right",
                f"{base_prompt}, different lighting"
            ]
            
            prompt = variation_prompts[i % len(variation_prompts)]
            avatar = self.generate_avatar(prompt, seed=base_seed + i)
            
            if avatar:
                avatars.append(avatar)
        
        return avatars
