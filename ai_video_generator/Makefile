# AI Video Generator Makefile

.PHONY: help install download-models run-web run-cli test clean

# Default target
help:
	@echo "AI Video Generator - Available Commands:"
	@echo ""
	@echo "  install         Install Python dependencies"
	@echo "  download-models Download required AI models"
	@echo "  run-web         Start web interface"
	@echo "  run-cli         Show CLI help"
	@echo "  test            Run tests"
	@echo "  clean           Clean temporary files"
	@echo "  setup           Complete setup (install + download models)"
	@echo ""

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Download AI models
download-models:
	@echo "Downloading AI models..."
	python download_models.py
	@echo "✅ Models downloaded successfully!"

# Complete setup
setup: install download-models
	@echo "🎉 Setup complete! You can now use the AI Video Generator."

# Start web interface
run-web:
	@echo "Starting web interface..."
	python app.py

# Show CLI help
run-cli:
	@echo "AI Video Generator CLI:"
	python cli.py --help

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "temp_*" -delete
	find . -type f -name "batch_*" -delete
	rm -rf uploads/*
	rm -rf outputs/*
	@echo "✅ Cleanup complete!"

# Create necessary directories
setup-dirs:
	@echo "Creating necessary directories..."
	mkdir -p models/tts
	mkdir -p models/avatar
	mkdir -p models/video
	mkdir -p models/talking_head
	mkdir -p models/face_detection
	mkdir -p models/background
	mkdir -p uploads
	mkdir -p outputs
	mkdir -p logs
	mkdir -p static
	mkdir -p templates
	@echo "✅ Directories created!"

# Check system requirements
check-requirements:
	@echo "Checking system requirements..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
	@python -c "import PIL; print(f'Pillow: {PIL.__version__}')"
	@echo "✅ System requirements checked!"

# Generate sample video
demo:
	@echo "Generating sample video..."
	python cli.py text-to-video -t "Welcome to AI Video Generator! This is a demo video created using local AI models." -o demo_video.mp4
	@echo "✅ Demo video generated: demo_video.mp4"

# Generate sample avatar
demo-avatar:
	@echo "Generating sample avatar..."
	python cli.py generate-avatar -p "professional businesswoman" -o demo_avatar.png
	@echo "✅ Demo avatar generated: demo_avatar.png"

# Generate sample audio
demo-audio:
	@echo "Generating sample audio..."
	python cli.py text-to-speech -t "This is a sample audio generated using AI text-to-speech." -o demo_audio.wav
	@echo "✅ Demo audio generated: demo_audio.wav"

# Full demo
full-demo: demo-avatar demo-audio demo
	@echo "🎬 Full demo completed! Check the generated files."

# Docker build (if Dockerfile exists)
docker-build:
	@echo "Building Docker image..."
	docker build -t ai-video-generator .
	@echo "✅ Docker image built successfully!"

# Docker run
docker-run:
	@echo "Running Docker container..."
	docker run -p 5000:5000 -v $(PWD)/outputs:/app/outputs ai-video-generator
	@echo "✅ Docker container running!"

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated!"

# Check for updates
check-updates:
	@echo "Checking for updates..."
	pip list --outdated
	@echo "✅ Update check complete!"

# Backup models
backup-models:
	@echo "Backing up models..."
	tar -czf models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/
	@echo "✅ Models backed up!"

# Restore models
restore-models:
	@echo "Restoring models..."
	@echo "Please specify the backup file: make restore-models BACKUP=filename.tar.gz"
	@if [ -n "$(BACKUP)" ]; then tar -xzf $(BACKUP); echo "✅ Models restored from $(BACKUP)!"; else echo "❌ Please specify BACKUP=filename.tar.gz"; fi

# System info
info:
	@echo "AI Video Generator System Information:"
	@echo "======================================"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "OS: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@echo "Available disk space: $(shell df -h . | tail -1 | awk '{print $$4}')"
	@echo "Memory: $(shell free -h | grep Mem | awk '{print $$2}')"
	@echo "GPU: $(shell nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU detected')"
	@echo "======================================"
