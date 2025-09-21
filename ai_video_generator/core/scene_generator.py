"""
Scene Generator for AI Video Generator
Simplified version for testing without heavy ML dependencies
"""

import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import json
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class SceneGenerator:
    """Simplified scene generator for testing"""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cpu'  # Simplified for testing
        self.pipeline = None
        self.scene_templates = {}
        
    def load_model(self) -> bool:
        """Mock model loading for testing"""
        try:
            logger.info("Mock loading video generation model")
            # In a real implementation, this would load Stable Video Diffusion
            self.pipeline = {'loaded': True}
            logger.info("Mock video generation model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load video generation model: {e}")
            return False
    
    def parse_script(self, script_path: str) -> List[Dict]:
        """Parse a script file into scenes"""
        try:
            logger.info(f"Mock parsing script: {script_path}")
            
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse different script formats
            if script_path.endswith('.json'):
                return self._parse_json_script(content)
            elif script_path.endswith('.md'):
                return self._parse_markdown_script(content)
            else:
                return self._parse_text_script(content)
                
        except Exception as e:
            logger.error(f"Failed to parse script: {e}")
            return []
    
    def _parse_json_script(self, content: str) -> List[Dict]:
        """Parse JSON script format"""
        try:
            data = json.loads(content)
            return data.get('scenes', [])
        except Exception as e:
            logger.error(f"Failed to parse JSON script: {e}")
            return []
    
    def _parse_markdown_script(self, content: str) -> List[Dict]:
        """Parse Markdown script format"""
        scenes = []
        lines = content.split('\n')
        current_scene = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('## Scene'):
                if current_scene:
                    scenes.append(current_scene)
                current_scene = {
                    'title': line,
                    'text': '',
                    'duration': 5,
                    'style': 'default'
                }
            elif line.startswith('**Duration:**'):
                if current_scene:
                    duration = re.findall(r'\d+', line)
                    if duration:
                        current_scene['duration'] = int(duration[0])
            elif line.startswith('**Style:**'):
                if current_scene:
                    style = line.replace('**Style:**', '').strip()
                    current_scene['style'] = style
            elif line and not line.startswith('#'):
                if current_scene:
                    current_scene['text'] += line + ' '
        
        if current_scene:
            scenes.append(current_scene)
        
        return scenes
    
    def _parse_text_script(self, content: str) -> List[Dict]:
        """Parse plain text script format"""
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        scenes = []
        for i, paragraph in enumerate(paragraphs):
            scene = {
                'title': f"Scene {i + 1}",
                'text': paragraph,
                'duration': 5,
                'style': 'default'
            }
            scenes.append(scene)
        
        return scenes
    
    def generate_scene(self, scene_data: Dict) -> Optional[str]:
        """Mock scene generation"""
        try:
            logger.info(f"Mock generating scene: {scene_data.get('title', 'Unknown')}")
            
            if self.pipeline is None:
                if not self.load_model():
                    return None
            
            # Create mock scene image
            initial_image = self._create_scene_image(scene_data)
            
            # Mock video generation - just create a text file for testing
            output_path = f"scene_{scene_data.get('title', 'unknown').replace(' ', '_')}.mp4"
            with open(output_path, 'w') as f:
                f.write(f"# Mock scene video\n")
                f.write(f"# Title: {scene_data.get('title', 'Unknown')}\n")
                f.write(f"# Text: {scene_data.get('text', '')[:100]}...\n")
                f.write(f"# Duration: {scene_data.get('duration', 5)}s\n")
                f.write(f"# Style: {scene_data.get('style', 'default')}\n")
            
            logger.info(f"Mock scene generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate scene: {e}")
            return None
    
    def _create_scene_image(self, scene_data: Dict) -> Image.Image:
        """Create mock scene image"""
        # Get scene dimensions
        width, height = self.config.get('resolution', [1024, 576])
        
        # Create base image
        image = Image.new('RGB', (width, height), color='#f0f0f0')
        draw = ImageDraw.Draw(image)
        
        # Add text overlay
        text = scene_data.get('text', '')
        style = scene_data.get('style', 'default')
        
        # Apply style
        if style == 'professional':
            bg_color = '#ffffff'
            text_color = '#333333'
            font_size = 32
        elif style == 'modern':
            bg_color = '#1a1a1a'
            text_color = '#ffffff'
            font_size = 36
        elif style == 'creative':
            bg_color = '#667eea'
            text_color = '#ffffff'
            font_size = 34
        else:
            bg_color = '#f8f9fa'
            text_color = '#2c3e50'
            font_size = 30
        
        # Fill background
        draw.rectangle([0, 0, width, height], fill=bg_color)
        
        # Add text
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Wrap text
        wrapped_text = self._wrap_text(text, font, width - 100)
        
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text with shadow
        draw.text((x+2, y+2), wrapped_text, font=font, fill='#000000')
        draw.text((x, y), wrapped_text, font=font, fill=text_color)
        
        return image
    
    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _save_video_frames(self, frames: List[Image.Image], output_path: str):
        """Mock saving video frames"""
        logger.info(f"Mock saving {len(frames)} frames to {output_path}")
        # In a real implementation, this would save actual video frames
        with open(output_path, 'w') as f:
            f.write(f"# Mock video with {len(frames)} frames\n")

class AdvancedSceneGenerator(SceneGenerator):
    """Advanced scene generator with mock features"""
    
    def __init__(self, config):
        super().__init__(config)
        self.transition_effects = {}
        self.background_generator = None
    
    def load_background_generator(self):
        """Mock background generator loading"""
        try:
            logger.info("Mock loading background generator")
            self.background_generator = {'loaded': True}
            logger.info("Mock background generator loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load background generator: {e}")
            return False
    
    def generate_scene_with_background(self, scene_data: Dict) -> Optional[str]:
        """Mock scene generation with background"""
        try:
            logger.info("Mock generating scene with background")
            return self.generate_scene(scene_data)
        except Exception as e:
            logger.error(f"Failed to generate scene with background: {e}")
            return None
    
    def _generate_background(self, prompt: str) -> Image.Image:
        """Mock background generation"""
        try:
            logger.info(f"Mock generating background: {prompt}")
            width, height = self.config.get('resolution', [1024, 576])
            image = Image.new('RGB', (width, height), color='#e0e0e0')
            return image
        except Exception as e:
            logger.error(f"Failed to generate background: {e}")
            return None
    
    def _create_scene_with_background(self, scene_data: Dict, background: Image.Image) -> Image.Image:
        """Mock scene creation with background"""
        # Start with background
        scene_image = background.copy()
        draw = ImageDraw.Draw(scene_image)
        
        # Add semi-transparent overlay
        overlay = Image.new('RGBA', scene_image.size, (0, 0, 0, 100))
        scene_image = Image.alpha_composite(scene_image.convert('RGBA'), overlay).convert('RGB')
        
        # Add text
        text = scene_data.get('text', '')
        style = scene_data.get('style', 'default')
        
        # Apply style
        if style == 'professional':
            text_color = '#ffffff'
            font_size = 32
        elif style == 'modern':
            text_color = '#ffffff'
            font_size = 36
        else:
            text_color = '#ffffff'
            font_size = 30
        
        # Add text with background
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Wrap text
        wrapped_text = self._wrap_text(text, font, scene_image.width - 100)
        
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (scene_image.width - text_width) // 2
        y = (scene_image.height - text_height) // 2
        
        # Draw text background
        padding = 20
        draw.rectangle([x-padding, y-padding, x+text_width+padding, y+text_height+padding], 
                      fill=(0, 0, 0, 150))
        
        # Draw text
        draw.text((x, y), wrapped_text, font=font, fill=text_color)
        
        return scene_image