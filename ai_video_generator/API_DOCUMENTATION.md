# 📚 AI Video Generator - API Documentation

## Overview

The AI Video Generator provides both REST API endpoints and Python API for creating videos from text content using local AI models.

## REST API Endpoints

### Base URL
```
http://localhost:5000
```

### Authentication
Currently, no authentication is required. All endpoints are publicly accessible.

---

## Video Generation

### POST /api/generate
Generate video from text content.

**Request Body:**
```json
{
  "text": "Your text content here",
  "style": "professional",
  "duration": 10,
  "voice": "female_professional",
  "effects": ["zoom", "fade"]
}
```

**Parameters:**
- `text` (string, required): Text content to convert to video
- `style` (string, optional): Video style - "professional", "modern", "creative"
- `duration` (integer, optional): Video duration in seconds (5-60)
- `voice` (string, optional): Voice for text-to-speech
- `effects` (array, optional): Special effects to apply

**Response:**
```json
{
  "success": true,
  "video_url": "/download/generated_video.mp4",
  "message": "Video generated successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to AI Video Generator!",
    "style": "professional",
    "duration": 10
  }'
```

### POST /api/generate-from-script
Generate video from uploaded script file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: File upload (supports .txt, .md, .json)

**Response:**
```json
{
  "success": true,
  "video_url": "/download/generated_video.mp4",
  "message": "Video generated from script successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/generate-from-script \
  -F "file=@script.md"
```

---

## Audio Generation

### POST /api/preview-audio
Generate audio preview from text.

**Request Body:**
```json
{
  "text": "Your text here",
  "voice": "female_professional"
}
```

**Response:**
```json
{
  "success": true,
  "audio_url": "/download/preview_audio.wav",
  "message": "Audio preview generated successfully"
}
```

---

## Avatar Generation

### POST /api/avatar
Generate avatar image.

**Request Body:**
```json
{
  "prompt": "professional businesswoman",
  "style": "realistic"
}
```

**Parameters:**
- `prompt` (string, required): Description of the avatar
- `style` (string, optional): Style - "realistic", "cartoon", "anime", "professional", "casual"

**Response:**
```json
{
  "success": true,
  "avatar_url": "/download/generated_avatar.png",
  "message": "Avatar generated successfully"
}
```

---

## System Information

### GET /api/voices
Get available voices for text-to-speech.

**Response:**
```json
{
  "voices": [
    "female_professional",
    "male_deep",
    "female_young"
  ]
}
```

### GET /api/status
Get system status and component availability.

**Response:**
```json
{
  "tts_engine": "loaded",
  "avatar_generator": "loaded",
  "talking_head": "loaded",
  "scene_generator": "loaded",
  "video_composer": "loaded"
}
```

### GET /api/templates
Get available video templates.

**Response:**
```json
{
  "templates": {
    "professional": {
      "background": "#ffffff",
      "text_color": "#333333",
      "font_family": "Arial",
      "font_size": 24,
      "layout": "center"
    },
    "modern": {
      "background": "#1a1a1a",
      "text_color": "#ffffff",
      "font_family": "Helvetica",
      "font_size": 28,
      "layout": "left"
    }
  }
}
```

---

## File Downloads

### GET /download/{filename}
Download generated files (videos, audio, images).

**Parameters:**
- `filename` (string): Name of the file to download

**Response:**
- File download (binary content)

---

## Error Handling

### Error Response Format
```json
{
  "error": "Error message description"
}
```

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (file not found)
- `413`: Payload Too Large (file too big)
- `500`: Internal Server Error

### Common Error Messages
- `"Text is required"`: Missing text parameter
- `"No file uploaded"`: Missing file in upload request
- `"Failed to generate video"`: Video generation failed
- `"File not found"`: Requested file doesn't exist
- `"File too large"`: Uploaded file exceeds size limit

---

## Python API

### Core Classes

#### VideoComposer
Main class for video generation.

```python
from core.video_composer import VideoComposer
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create composer
composer = VideoComposer(config)

# Generate video from text
video_path = composer.create_from_text(
    text="Hello, world!",
    style="professional",
    duration=10,
    voice="female_professional"
)

# Generate video from script
video_path = composer.create_from_script("script.json")

# Add effects
video_path = composer.create_with_effects(
    text="Hello, world!",
    effects=["zoom", "fade"]
)
```

#### TTSEngine
Text-to-speech functionality.

```python
from core.tts_engine import TTSEngine

# Initialize TTS engine
tts = TTSEngine(config['models']['tts'])

# Generate speech
audio_path = tts.generate_speech(
    text="Hello, world!",
    voice="female_professional"
)

# Get available voices
voices = tts.get_available_voices()

# Generate batch audio
texts = ["Hello", "World", "AI Video"]
audio_paths = tts.generate_batch(texts, "female_professional")
```

#### AvatarGenerator
Avatar generation functionality.

```python
from core.avatar_generator import AvatarGenerator

# Initialize avatar generator
avatar_gen = AvatarGenerator(config['models']['avatar'])

# Generate avatar
avatar = avatar_gen.generate_avatar(
    prompt="professional person",
    style="realistic"
)

# Generate with background
avatar_with_bg = avatar_gen.create_avatar_with_background(
    avatar, "professional"
)

# Add text overlay
avatar_with_text = avatar_gen.add_text_overlay(
    avatar, "John Doe", "bottom"
)
```

#### TalkingHead
Talking head animation.

```python
from core.talking_head import TalkingHead
from PIL import Image

# Initialize talking head
talking_head = TalkingHead(config['models']['talking_head'])

# Load avatar image
avatar = Image.open("avatar.png")

# Create talking head video
video_path = talking_head.create_talking_head(
    avatar_image=avatar,
    audio_path="speech.wav"
)
```

#### SceneGenerator
Video scene generation.

```python
from core.scene_generator import SceneGenerator

# Initialize scene generator
scene_gen = SceneGenerator(config['models']['video'])

# Parse script
scenes = scene_gen.parse_script("script.json")

# Generate scene
scene_video = scene_gen.generate_scene(scenes[0])

# Generate with background
scene_with_bg = scene_gen.generate_scene_with_background(scenes[0])
```

---

## Configuration

### Configuration File (config.yaml)

```yaml
# Model settings
models:
  tts:
    default: "tts_models/en/ljspeech/tacotron2-DDC"
    voices:
      female_professional:
        model: "tts_models/en/ljspeech/tacotron2-DDC"
        speaker: "default"
  
  avatar:
    default: "stabilityai/stable-diffusion-xl-base-1.0"
    style: "realistic"
    resolution: [512, 512]
  
  talking_head:
    model: "wav2lip"
    checkpoint: "models/talking_head/wav2lip_gan.pth"
  
  video:
    model: "stabilityai/stable-video-diffusion"
    resolution: [1024, 576]
    fps: 24

# Processing settings
processing:
  device: "auto"  # auto, cpu, cuda
  batch_size: 1
  max_workers: 4
  memory_limit: "8GB"

# Output settings
output:
  format: "mp4"
  quality: "high"
  fps: 24
  resolution: [1920, 1080]
  bitrate: "5000k"

# Web interface
web:
  host: "0.0.0.0"
  port: 5000
  debug: false
  upload_folder: "uploads"
  output_folder: "outputs"
  max_file_size: "100MB"
```

---

## Examples

### Complete Video Generation Workflow

```python
import yaml
from core.video_composer import VideoComposer
from core.tts_engine import TTSEngine
from core.avatar_generator import AvatarGenerator

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
composer = VideoComposer(config)
tts = TTSEngine(config['models']['tts'])
avatar_gen = AvatarGenerator(config['models']['avatar'])

# Generate avatar
avatar = avatar_gen.generate_avatar("professional businesswoman")

# Generate speech
audio_path = tts.generate_speech("Welcome to our presentation!")

# Generate video
video_path = composer.create_from_text(
    text="Welcome to our presentation!",
    style="professional",
    duration=15,
    voice="female_professional"
)

print(f"Video generated: {video_path}")
```

### Batch Processing

```python
# Process multiple texts
texts = [
    "Introduction to AI",
    "Machine Learning Basics",
    "Deep Learning Applications"
]

videos = []
for i, text in enumerate(texts):
    video_path = composer.create_from_text(
        text=text,
        style="professional",
        duration=10
    )
    videos.append(video_path)
    print(f"Generated video {i+1}: {video_path}")
```

### Custom Script Processing

```python
# Parse custom script format
script_data = {
    "title": "My Presentation",
    "scenes": [
        {
            "title": "Introduction",
            "text": "Welcome to my presentation",
            "duration": 8,
            "style": "professional"
        },
        {
            "title": "Main Content",
            "text": "This is the main content",
            "duration": 15,
            "style": "modern"
        }
    ]
}

# Generate video from script
video_path = composer.create_from_script_data(script_data)
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing:

- Request rate limiting
- File size limits
- Processing queue management
- Resource usage monitoring

---

## Monitoring and Logging

### Log Files
- Application logs: `logs/ai_video_generator.log`
- Error logs: Console output
- Model logs: Individual model directories

### Health Checks
```bash
# Check system status
curl http://localhost:5000/api/status

# Check specific component
curl http://localhost:5000/api/voices
```

---

## Security Considerations

- All processing happens locally
- No external API calls for core functionality
- File uploads are validated
- Generated content is stored locally
- Consider implementing authentication for production use

---

## Performance Tips

1. **Use GPU acceleration** when available
2. **Optimize batch sizes** based on available memory
3. **Pre-download models** to avoid delays
4. **Use appropriate video quality** settings
5. **Clean up temporary files** regularly

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model download fails**: Check internet connection
3. **Video generation fails**: Verify FFmpeg installation
4. **Audio issues**: Check audio dependencies

### Debug Mode

Enable debug mode in config.yaml:
```yaml
web:
  debug: true
```

Or start with debug flag:
```bash
python app.py --debug
```
