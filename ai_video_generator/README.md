# 🎬 Local AI Video Generator

A comprehensive local AI video generation tool that combines multiple approaches for creating videos from text content, similar to Synthesia, D-ID, Lumen5, and Pictory.

## 🌟 Features

- **Text-to-Speech (TTS)** with multiple voice options
- **AI Avatar Generation** using local models
- **Talking Head Animation** with lip-sync
- **Text-to-Video** with scene generation
- **Script-to-Video** with automatic scene creation
- **Multiple Export Formats** (MP4, WebM, GIF)
- **Web Interface** for easy interaction
- **Batch Processing** for multiple videos
- **Customizable Templates** and styles

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download models (first time only)
python download_models.py

# Start the web interface
python app.py

# Or use CLI
python generate_video.py --input script.txt --output video.mp4
```

## 📁 Project Structure

```
ai_video_generator/
├── app.py                 # Web interface
├── cli.py                 # Command line interface
├── core/
│   ├── tts_engine.py      # Text-to-speech
│   ├── avatar_generator.py # AI avatar creation
│   ├── talking_head.py    # Talking head animation
│   ├── scene_generator.py # Video scene generation
│   └── video_composer.py  # Final video assembly
├── models/
│   ├── tts/              # TTS models
│   ├── avatar/           # Avatar generation models
│   └── video/            # Video generation models
├── templates/
│   ├── styles/           # Visual templates
│   └── layouts/          # Layout configurations
├── static/               # Web interface assets
├── examples/             # Sample scripts and outputs
└── requirements.txt      # Python dependencies
```

## 🎯 Usage Examples

### 1. Text-to-Video
```python
from core.video_composer import VideoComposer

composer = VideoComposer()
video = composer.create_from_text(
    text="Welcome to our AI video generator!",
    style="professional",
    duration=10
)
```

### 2. Script-to-Video
```python
from core.scene_generator import SceneGenerator

generator = SceneGenerator()
scenes = generator.parse_script("script.md")
video = generator.create_video(scenes)
```

### 3. Talking Head
```python
from core.talking_head import TalkingHead

avatar = TalkingHead()
video = avatar.create_talking_head(
    text="Hello, I'm your AI assistant!",
    voice="female_professional"
)
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model paths and settings
- Voice options and languages
- Visual styles and templates
- Output quality and format
- Processing parameters

## 📊 Supported Formats

**Input:**
- Plain text files
- Markdown documents
- JSON scripts
- CSV data

**Output:**
- MP4 (H.264)
- WebM (VP9)
- GIF animations
- Image sequences

## 🌐 Web Interface

Access the web interface at `http://localhost:5000` to:
- Upload text files
- Preview generated content
- Customize settings
- Download videos
- Manage templates

## 🎨 Customization

- **Templates**: Create custom visual styles
- **Voices**: Add new TTS voices
- **Avatars**: Train custom avatar models
- **Scenes**: Design new scene layouts

## 📈 Performance

- **GPU Acceleration**: CUDA support for faster processing
- **Batch Processing**: Generate multiple videos simultaneously
- **Memory Optimization**: Efficient model loading and caching
- **Progress Tracking**: Real-time generation progress

## 🔒 Privacy

- **100% Local**: No data sent to external services
- **Offline Capable**: Works without internet connection
- **Data Control**: Complete control over your content
- **Secure**: All processing happens on your machine

## 🛠️ Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space for models

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Manual](docs/user_manual.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)
