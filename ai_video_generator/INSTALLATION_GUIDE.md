---
# Auto-generated front matter
Title: Installation Guide
LastUpdated: 2025-11-06T20:45:57.737815
Tags: []
Status: draft
---

# 🚀 AI Video Generator - Installation Guide

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Hardware Recommendations
- **CPU**: 4+ cores, 2.5GHz+
- **GPU**: NVIDIA RTX 3060 or better (for faster processing)
- **RAM**: 16GB+ for smooth operation
- **Storage**: SSD recommended for faster model loading

## Installation Methods

### Method 1: Quick Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ai_video_generator

# Run the setup script
make setup
```

### Method 2: Manual Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download AI models
python download_models.py

# 3. Create necessary directories
make setup-dirs
```

### Method 3: Docker Installation

```bash
# Build and run with Docker
docker-compose up -d

# Or build manually
docker build -t ai-video-generator .
docker run -p 5000:5000 ai-video-generator
```

## Configuration

### 1. Edit Configuration File

Edit `config.yaml` to customize settings:

```yaml
# Device settings
processing:
  device: "auto"  # auto, cpu, cuda
  batch_size: 1
  memory_limit: "8GB"

# Output settings
output:
  format: "mp4"
  quality: "high"
  resolution: [1920, 1080]
  fps: 24
```

### 2. GPU Setup (Optional)

If you have an NVIDIA GPU:

```bash
# Install CUDA toolkit
# Follow NVIDIA's installation guide for your OS

# Verify CUDA installation
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Model Configuration

The system will automatically download models on first use. You can also pre-download them:

```bash
# Download all models
make download-models

# Or download specific models
python cli.py download-models
```

## Usage

### Web Interface

```bash
# Start web interface
make run-web
# or
python app.py

# Access at http://localhost:5000
```

### Command Line Interface

```bash
# Generate video from text
python cli.py text-to-video -t "Your text here" -o output.mp4

# Generate video from script
python cli.py script-to-video -s script.json -o output.mp4

# Generate avatar
python cli.py generate-avatar -p "professional person" -o avatar.png

# Generate speech
python cli.py text-to-speech -t "Your text" -o speech.wav

# Check system status
python cli.py status
```

### Python API

```python
from core.video_composer import VideoComposer
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create video composer
composer = VideoComposer(config)

# Generate video
video_path = composer.create_from_text(
    text="Hello, world!",
    style="professional",
    duration=10
)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in config.yaml
processing:
  batch_size: 1
  memory_limit: "4GB"
```

#### 2. Model Download Fails
```bash
# Check internet connection
# Try downloading models individually
python cli.py download-models
```

#### 3. Audio Generation Issues
```bash
# Install additional audio dependencies
pip install soundfile librosa
```

#### 4. Video Generation Fails
```bash
# Check FFmpeg installation
ffmpeg -version

# Install FFmpeg if missing
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

### Performance Optimization

#### 1. GPU Acceleration
- Ensure CUDA is properly installed
- Set `device: "cuda"` in config.yaml
- Use `nvidia-smi` to monitor GPU usage

#### 2. Memory Optimization
- Reduce `batch_size` in config.yaml
- Close other applications during video generation
- Use `memory_limit` setting

#### 3. Storage Optimization
- Use SSD for faster model loading
- Clean up temporary files regularly: `make clean`
- Compress output videos if needed

## Advanced Configuration

### Custom Models

You can use custom models by modifying `config.yaml`:

```yaml
models:
  tts:
    default: "your_custom_tts_model"
  avatar:
    default: "your_custom_avatar_model"
  video:
    default: "your_custom_video_model"
```

### Custom Templates

Create custom templates in `templates/` directory:

```yaml
templates:
  custom_style:
    background: "#your_color"
    text_color: "#your_color"
    font_family: "Your Font"
    font_size: 24
    layout: "your_layout"
```

### Environment Variables

Set environment variables for configuration:

```bash
export AI_VIDEO_DEVICE=cuda
export AI_VIDEO_MEMORY_LIMIT=8GB
export AI_VIDEO_OUTPUT_QUALITY=high
```

## Testing Installation

### 1. Run System Check

```bash
python cli.py status
```

### 2. Generate Test Video

```bash
python cli.py text-to-video -t "Test video" -o test.mp4
```

### 3. Run Demo

```bash
make full-demo
```

## Support

### Getting Help

1. Check the troubleshooting section above
2. Review the logs in `logs/` directory
3. Check system requirements
4. Verify all dependencies are installed

### Logs

- Application logs: `logs/ai_video_generator.log`
- Error logs: Check console output
- Model logs: Check individual model directories

### Performance Monitoring

```bash
# Check system resources
make info

# Monitor GPU usage (if available)
nvidia-smi

# Check disk usage
du -sh models/
```

## Updates

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Update dependencies
make update-deps

# Re-download models if needed
make download-models
```

### Updating Models

```bash
# Update specific models
python cli.py download-models

# Clean old models
make clean
make download-models
```

## Security Notes

- All processing happens locally
- No data is sent to external services
- Models are downloaded from official sources
- Generated content is stored locally

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Changelog

See CHANGELOG.md for version history and updates.
