"""
Video Composer for AI Video Generator
Combines all components to create final videos
"""

import cv2
import numpy as np
from PIL import Image
import moviepy.editor as mp
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

class VideoComposer:
    """Main video composer that combines all components"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.temp_files = []
        
    def create_from_text(self, text: str, style: str = "professional", 
                        duration: int = 10, voice: str = None) -> Optional[str]:
        """Create video from text content"""
        try:
            # Import components
            from .tts_engine import TTSEngine
            from .avatar_generator import AvatarGenerator
            from .talking_head import TalkingHead
            from .scene_generator import SceneGenerator
            
            # Initialize components
            tts = TTSEngine(self.config['models']['tts'])
            avatar_gen = AvatarGenerator(self.config['models']['avatar'])
            talking_head = TalkingHead(self.config['models']['talking_head'])
            scene_gen = SceneGenerator(self.config['models']['video'])
            
            # Generate audio
            logger.info("Generating speech...")
            audio_path = tts.generate_speech(text, voice)
            if not audio_path:
                return None
            
            # Generate avatar
            logger.info("Generating avatar...")
            avatar = avatar_gen.generate_avatar("professional person, business attire")
            if not avatar:
                return None
            
            # Create talking head
            logger.info("Creating talking head...")
            talking_head_path = talking_head.create_talking_head(avatar, audio_path)
            if not talking_head_path:
                return None
            
            # Generate background scenes
            logger.info("Generating background scenes...")
            scene_data = {
                'title': 'Main Scene',
                'text': text,
                'duration': duration,
                'style': style
            }
            scene_path = scene_gen.generate_scene(scene_data)
            if not scene_path:
                return None
            
            # Compose final video
            logger.info("Composing final video...")
            final_video = self._compose_video(talking_head_path, scene_path, audio_path)
            
            # Cleanup temp files
            self._cleanup_temp_files([audio_path, talking_head_path, scene_path])
            
            return final_video
            
        except Exception as e:
            logger.error(f"Failed to create video from text: {e}")
            return None
    
    def create_from_script(self, script_path: str) -> Optional[str]:
        """Create video from script file"""
        try:
            from .scene_generator import SceneGenerator
            from .tts_engine import TTSEngine
            from .avatar_generator import AvatarGenerator
            from .talking_head import TalkingHead
            
            # Initialize components
            scene_gen = SceneGenerator(self.config['models']['video'])
            tts = TTSEngine(self.config['models']['tts'])
            avatar_gen = AvatarGenerator(self.config['models']['avatar'])
            talking_head = TalkingHead(self.config['models']['talking_head'])
            
            # Parse script
            scenes = scene_gen.parse_script(script_path)
            if not scenes:
                return None
            
            # Generate video for each scene
            scene_videos = []
            for i, scene in enumerate(scenes):
                logger.info(f"Processing scene {i+1}/{len(scenes)}: {scene.get('title', 'Unknown')}")
                
                # Generate audio
                audio_path = tts.generate_speech(scene['text'])
                if not audio_path:
                    continue
                
                # Generate avatar
                avatar = avatar_gen.generate_avatar("professional person")
                if not avatar:
                    continue
                
                # Create talking head
                talking_head_path = talking_head.create_talking_head(avatar, audio_path)
                if not talking_head_path:
                    continue
                
                # Generate scene
                scene_video_path = scene_gen.generate_scene(scene)
                if not scene_video_path:
                    continue
                
                # Compose scene
                composed_scene = self._compose_video(talking_head_path, scene_video_path, audio_path)
                if composed_scene:
                    scene_videos.append(composed_scene)
                
                # Cleanup temp files
                self._cleanup_temp_files([audio_path, talking_head_path, scene_video_path])
            
            if not scene_videos:
                return None
            
            # Combine all scenes
            final_video = self._combine_scenes(scene_videos)
            
            # Cleanup scene videos
            self._cleanup_temp_files(scene_videos)
            
            return final_video
            
        except Exception as e:
            logger.error(f"Failed to create video from script: {e}")
            return None
    
    def _compose_video(self, talking_head_path: str, scene_path: str, 
                      audio_path: str) -> Optional[str]:
        """Compose talking head with background scene"""
        try:
            # Load videos
            talking_head_video = mp.VideoFileClip(talking_head_path)
            scene_video = mp.VideoFileClip(scene_path)
            audio = mp.AudioFileClip(audio_path)
            
            # Resize talking head to fit in corner
            talking_head_video = talking_head_video.resize(height=300)
            
            # Position talking head in bottom right corner
            talking_head_video = talking_head_video.set_position(('right', 'bottom'))
            
            # Composite videos
            final_video = mp.CompositeVideoClip([scene_video, talking_head_video])
            
            # Set audio
            final_video = final_video.set_audio(audio)
            
            # Set duration
            final_video = final_video.set_duration(min(talking_head_video.duration, 
                                                     scene_video.duration, 
                                                     audio.duration))
            
            # Save final video
            output_path = "composed_video.mp4"
            final_video.write_videofile(
                output_path,
                fps=25,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Close clips
            talking_head_video.close()
            scene_video.close()
            audio.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to compose video: {e}")
            return None
    
    def _combine_scenes(self, scene_videos: List[str]) -> Optional[str]:
        """Combine multiple scene videos into one"""
        try:
            if not scene_videos:
                return None
            
            if len(scene_videos) == 1:
                return scene_videos[0]
            
            # Load all videos
            clips = [mp.VideoFileClip(video) for video in scene_videos]
            
            # Concatenate videos
            final_video = mp.concatenate_videoclips(clips)
            
            # Save combined video
            output_path = "combined_video.mp4"
            final_video.write_videofile(
                output_path,
                fps=25,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Close clips
            for clip in clips:
                clip.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to combine scenes: {e}")
            return None
    
    def add_transitions(self, video_path: str, transition_type: str = "fade") -> Optional[str]:
        """Add transitions between scenes"""
        try:
            video = mp.VideoFileClip(video_path)
            
            if transition_type == "fade":
                # Add fade in/out
                video = video.fadein(1).fadeout(1)
            elif transition_type == "slide":
                # Add slide transition (simplified)
                video = video.fadein(0.5).fadeout(0.5)
            
            # Save with transitions
            output_path = video_path.replace('.mp4', '_with_transitions.mp4')
            video.write_videofile(
                output_path,
                fps=25,
                codec='libx264',
                audio_codec='aac'
            )
            
            video.close()
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to add transitions: {e}")
            return video_path
    
    def add_subtitles(self, video_path: str, text: str) -> Optional[str]:
        """Add subtitles to video"""
        try:
            video = mp.VideoFileClip(video_path)
            
            # Create subtitle clip
            subtitle = mp.TextClip(
                text,
                fontsize=24,
                color='white',
                font='Arial',
                stroke_color='black',
                stroke_width=2
            ).set_position(('center', 'bottom')).set_duration(video.duration)
            
            # Composite with subtitles
            final_video = mp.CompositeVideoClip([video, subtitle])
            
            # Save with subtitles
            output_path = video_path.replace('.mp4', '_with_subtitles.mp4')
            final_video.write_videofile(
                output_path,
                fps=25,
                codec='libx264',
                audio_codec='aac'
            )
            
            video.close()
            subtitle.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to add subtitles: {e}")
            return video_path
    
    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")

class AdvancedVideoComposer(VideoComposer):
    """Advanced video composer with more features"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.effects = {}
        self.templates = {}
    
    def create_with_effects(self, text: str, effects: List[str] = None) -> Optional[str]:
        """Create video with special effects"""
        try:
            # Create base video
            base_video = self.create_from_text(text)
            if not base_video:
                return None
            
            # Apply effects
            current_video = base_video
            for effect in effects or []:
                if effect == "zoom":
                    current_video = self._apply_zoom_effect(current_video)
                elif effect == "pan":
                    current_video = self._apply_pan_effect(current_video)
                elif effect == "fade":
                    current_video = self.add_transitions(current_video, "fade")
            
            return current_video
            
        except Exception as e:
            logger.error(f"Failed to create video with effects: {e}")
            return None
    
    def _apply_zoom_effect(self, video_path: str) -> Optional[str]:
        """Apply zoom effect to video"""
        try:
            video = mp.VideoFileClip(video_path)
            
            # Create zoom effect
            def zoom_in(t):
                return 1 + 0.1 * t / video.duration
            
            zoomed_video = video.resize(zoom_in)
            
            # Save zoomed video
            output_path = video_path.replace('.mp4', '_zoomed.mp4')
            zoomed_video.write_videofile(
                output_path,
                fps=25,
                codec='libx264',
                audio_codec='aac'
            )
            
            video.close()
            zoomed_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to apply zoom effect: {e}")
            return video_path
    
    def _apply_pan_effect(self, video_path: str) -> Optional[str]:
        """Apply pan effect to video"""
        try:
            video = mp.VideoFileClip(video_path)
            
            # Create pan effect
            def pan_left(t):
                return ('left', 'center')
            
            panned_video = video.set_position(pan_left)
            
            # Save panned video
            output_path = video_path.replace('.mp4', '_panned.mp4')
            panned_video.write_videofile(
                output_path,
                fps=25,
                codec='libx264',
                audio_codec='aac'
            )
            
            video.close()
            panned_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to apply pan effect: {e}")
            return video_path
