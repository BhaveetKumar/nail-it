#!/usr/bin/env python3
"""
Comprehensive script to generate README files for every directory with proper links to all content inside.
"""

import os
import re
from pathlib import Path

def get_directory_contents(directory):
    """Get all files and subdirectories in a directory."""
    contents = {
        'files': [],
        'dirs': []
    }
    
    try:
        for item in os.listdir(directory):
            if item.startswith('.'):
                continue
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                contents['dirs'].append(item)
            elif item.endswith('.md'):
                contents['files'].append(item)
    except PermissionError:
        pass
    
    return contents

def get_file_description(file_path):
    """Get a description of a markdown file by reading its first few lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            for line in lines:
                if line.startswith('#') and len(line.strip()) > 1:
                    return line.strip('# ').strip()
                elif line.strip() and not line.startswith('#'):
                    return line.strip()[:100] + "..." if len(line.strip()) > 100 else line.strip()
    except:
        pass
    return "Documentation file"

def generate_readme_content(directory, relative_path=""):
    """Generate README content for a directory."""
    contents = get_directory_contents(directory)
    
    # Get directory name for title
    dir_name = os.path.basename(directory) if directory != '.' else 'Root'
    
    # Create title
    title = dir_name.replace('_', ' ').replace('-', ' ').title()
    
    # Generate content
    readme_content = f"# {title}\n\n"
    
    # Add description based on directory
    descriptions = {
        '01_fundamentals': "Core technical fundamentals for backend engineering interviews",
        '02_system_design': "System design patterns, architectures, and case studies",
        '03_backend_engineering': "Backend development patterns, APIs, and microservices",
        '04_devops_infrastructure': "DevOps practices, cloud platforms, and infrastructure",
        '05_ai_ml': "AI/ML concepts and implementations for backend systems",
        '06_behavioral': "Leadership skills, communication, and behavioral preparation",
        '07_company_specific': "Interview preparation materials for specific companies",
        '08_interview_prep': "Comprehensive interview preparation materials and practice",
        '09_curriculum': "Structured learning paths for different skill levels",
        '10_resources': "Additional resources, tools, and references",
        'algorithms': "Algorithms and data structures implementations",
        'advanced_algorithms': "Advanced algorithms including quantum computing and optimization",
        'programming': "Programming languages and implementation guides",
        'specialized_guides': "Specialized guides for different domains and technologies",
        'patterns': "System design patterns and architectural patterns",
        'architectures': "System architectures and design patterns",
        'case_studies': "Real-world system design case studies",
        'distributed_systems': "Distributed systems concepts and implementations",
        'realtime': "Real-time systems and streaming architectures",
        'scalability': "Scalability patterns and load balancing strategies",
        'api_design': "API design principles and best practices",
        'databases': "Database management and optimization",
        'microservices': "Microservices architecture and patterns",
        'message_queues': "Message queue patterns and implementations",
        'caching': "Caching strategies and implementations",
        'testing': "Testing strategies and best practices",
        'cloud': "Cloud platform services and architectures",
        'containers': "Containerization and orchestration",
        'ci_cd': "Continuous integration and deployment",
        'monitoring': "Monitoring, observability, and alerting",
        'security': "Security practices and implementations",
        'performance': "Performance engineering and optimization",
        'machine_learning': "Machine learning fundamentals and algorithms",
        'deep_learning': "Deep learning concepts and implementations",
        'mlops': "MLOps practices and model deployment",
        'backend_ml': "Backend systems for machine learning",
        'conflict_resolution': "Conflict resolution and team management",
        'razorpay': "Razorpay-specific interview preparation",
        'faang': "FAANG company interview patterns and preparation",
        'atlassian': "Atlassian interview preparation materials",
        'meta': "Meta (Facebook) interview preparation",
        'fintech': "Fintech and payment systems preparation",
        'other': "Other company-specific preparation materials",
        'shared': "Shared resources and common patterns",
        'guides': "Interview preparation guides and roadmaps",
        'practice': "Practice problems and coding challenges",
        'mock_interviews': "Mock interview scenarios and practice",
        'checklists': "Preparation checklists and templates",
        'phase0_fundamentals': "Phase 0: Fundamentals learning path",
        'phase1_intermediate': "Phase 1: Intermediate learning path",
        'phase2_advanced': "Phase 2: Advanced learning path",
        'phase3_expert': "Phase 3: Expert learning path",
        'reference': "Reference materials and quick guides",
        'tools': "Development tools and utilities",
        'external': "External resources and references",
        'progress_tracking': "Progress tracking and assessment tools",
        'projects': "Real-world projects and implementations"
    }
    
    if dir_name in descriptions:
        readme_content += f"{descriptions[dir_name]}\n\n"
    
    # Add table of contents
    readme_content += "## 📚 Table of Contents\n\n"
    
    # Add subdirectories
    if contents['dirs']:
        readme_content += "### 📁 Directories\n\n"
        for subdir in sorted(contents['dirs']):
            readme_content += f"- **[{subdir.replace('_', ' ').replace('-', ' ').title()}]({subdir}/)**\n"
        readme_content += "\n"
    
    # Add files
    if contents['files']:
        readme_content += "### 📄 Files\n\n"
        for file in sorted(contents['files']):
            if file != 'README.md':
                file_path = os.path.join(directory, file)
                description = get_file_description(file_path)
                readme_content += f"- **[{file.replace('.md', '').replace('_', ' ').replace('-', ' ').title()}]({file})** - {description}\n"
        readme_content += "\n"
    
    # Add quick start section
    readme_content += "## 🚀 Quick Start\n\n"
    
    if contents['dirs']:
        readme_content += "### Explore Subdirectories\n"
        for subdir in sorted(contents['dirs'])[:3]:  # Show first 3 directories
            readme_content += f"1. **[{subdir.replace('_', ' ').replace('-', ' ').title()}]({subdir}/)** - {descriptions.get(subdir, 'Explore this directory')}\n"
        readme_content += "\n"
    
    if contents['files']:
        readme_content += "### Key Files\n"
        for file in sorted(contents['files'])[:3]:  # Show first 3 files
            if file != 'README.md':
                readme_content += f"1. **[{file.replace('.md', '').replace('_', ' ').replace('-', ' ').title()}]({file})**\n"
        readme_content += "\n"
    
    # Add navigation
    readme_content += "## 🔗 Navigation\n\n"
    readme_content += "- **← Back to Parent**: [../](../)\n"
    readme_content += "- **🏠 Home**: [../../](../..)\n"
    readme_content += "- **📋 Master Index**: [../../MASTER_INDEX.md](../..MASTER_INDEX.md)\n\n"
    
    # Add study recommendations
    if dir_name in ['01_fundamentals', '02_system_design', '03_backend_engineering']:
        readme_content += "## 📖 Study Recommendations\n\n"
        readme_content += "1. **Start with fundamentals** if you're new to the topic\n"
        readme_content += "2. **Practice with examples** to reinforce learning\n"
        readme_content += "3. **Review patterns** and their use cases\n"
        readme_content += "4. **Implement solutions** to solidify understanding\n\n"
    
    # Add contribution note
    readme_content += "## 🤝 Contributing\n\n"
    readme_content += "Found an issue or want to add content? Please see our [Contributing Guidelines](../../CONTRIBUTING.md) for details.\n\n"
    
    # Add last updated
    readme_content += "---\n\n"
    readme_content += f"**Last Updated**: December 2024\n"
    readme_content += f"**Directory**: `{relative_path or '.'}`\n"
    
    return readme_content

def create_readme_for_directory(directory, relative_path=""):
    """Create README file for a directory."""
    readme_path = os.path.join(directory, 'README.md')
    
    # Skip if README already exists and is comprehensive
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 500:  # If README is already comprehensive, skip
                    return
        except:
            pass
    
    readme_content = generate_readme_content(directory, relative_path)
    
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ Created README for {relative_path or '.'}")
    except Exception as e:
        print(f"❌ Error creating README for {relative_path or '.'}: {e}")

def main():
    """Main function to generate README files for all directories."""
    print("🚀 Starting comprehensive README generation for all directories...")
    
    # Get all directories
    all_dirs = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        all_dirs.append(root)
    
    # Sort directories by depth and name
    all_dirs.sort(key=lambda x: (x.count(os.sep), x))
    
    created_count = 0
    skipped_count = 0
    
    for directory in all_dirs:
        if directory == '.':
            continue
            
        relative_path = directory[2:] if directory.startswith('./') else directory
        
        # Skip certain directories
        skip_dirs = ['.git', '.cursor-autocontinue', 'node_modules']
        if any(skip_dir in directory for skip_dir in skip_dirs):
            continue
        
        # Check if directory has content
        contents = get_directory_contents(directory)
        if not contents['dirs'] and not contents['files']:
            continue
        
        # Create README
        create_readme_for_directory(directory, relative_path)
        created_count += 1
    
    print(f"\n✅ README generation completed!")
    print(f"📊 Summary:")
    print(f"  - Created README files: {created_count}")
    print(f"  - Skipped directories: {skipped_count}")
    print(f"  - Total directories processed: {len(all_dirs)}")

if __name__ == "__main__":
    main()
