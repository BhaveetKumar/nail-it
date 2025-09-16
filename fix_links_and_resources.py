#!/usr/bin/env python3
"""
Comprehensive script to fix all links and add missing resources throughout the repository.
"""

import os
import re
import glob
from pathlib import Path

def find_all_md_files():
    """Find all markdown files in the repository."""
    md_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

def get_directory_structure():
    """Get the current directory structure."""
    structure = {}
    for root, dirs, files in os.walk('.'):
        if root.startswith('./.'):
            continue
        rel_path = root[2:] if root.startswith('./') else root
        if rel_path:
            structure[rel_path] = {
                'dirs': [d for d in dirs if not d.startswith('.')],
                'files': [f for f in files if f.endswith('.md')]
            }
    return structure

def create_missing_readmes():
    """Create README files for directories that don't have them."""
    structure = get_directory_structure()
    
    readme_templates = {
        '01_fundamentals': """# Fundamentals

Core technical fundamentals for backend engineering interviews.

## Contents
- [Algorithms & Data Structures](algorithms/)
- [Advanced Algorithms](advanced_algorithms/)
- [Programming Languages](programming/)
- [Computer Science](networking/)
- [Mathematics](mathematics/)

## Quick Start
1. Start with algorithms and data structures
2. Move to advanced algorithms
3. Practice with programming implementations
4. Review computer science fundamentals
""",
        
        '02_system_design': """# System Design

System design patterns, architectures, and case studies.

## Contents
- [Design Patterns](patterns/)
- [Architectures](architectures/)
- [Case Studies](case_studies/)
- [Distributed Systems](distributed_systems/)

## Quick Start
1. Learn basic design patterns
2. Study architecture patterns
3. Practice with case studies
4. Master distributed systems
""",
        
        '03_backend_engineering': """# Backend Engineering

Backend development patterns, APIs, and microservices.

## Contents
- [API Design](api_design/)
- [Database Management](databases/)
- [Microservices](microservices/)
- [Message Queues](message_queues/)
- [Caching](caching/)

## Quick Start
1. Master API design principles
2. Learn database optimization
3. Understand microservices patterns
4. Practice with message queues
""",
        
        '04_devops_infrastructure': """# DevOps & Infrastructure

DevOps practices, cloud platforms, and infrastructure management.

## Contents
- [Cloud Platforms](cloud/)
- [Containers](containers/)
- [CI/CD](ci_cd/)
- [Monitoring](monitoring/)
- [Security](security/)

## Quick Start
1. Learn cloud platform basics
2. Master containerization
3. Set up CI/CD pipelines
4. Implement monitoring
""",
        
        '05_ai_ml': """# AI/ML for Backend Engineers

Machine learning concepts and implementations for backend systems.

## Contents
- [Machine Learning](machine_learning/)
- [Deep Learning](deep_learning/)
- [MLOps](mlops/)
- [Backend for AI](backend_for_ai/)

## Quick Start
1. Learn ML fundamentals
2. Understand deep learning
3. Master MLOps practices
4. Build AI backend systems
""",
        
        '06_behavioral': """# Behavioral & Leadership

Leadership skills, communication, and behavioral interview preparation.

## Contents
- [Leadership Skills](leadership_skills/)
- [Communication](communication/)
- [Team Management](team_management/)
- [Problem Solving](problem_solving/)

## Quick Start
1. Develop leadership skills
2. Practice communication
3. Learn team management
4. Master problem solving
""",
        
        '07_company_specific': """# Company-Specific Content

Interview preparation materials for specific companies.

## Contents
- [Razorpay](razorpay/)
- [FAANG+](faang/)
- [Other Companies](other/)
- [Fintech](fintech/)

## Quick Start
1. Choose your target company
2. Study company-specific content
3. Practice with mock interviews
4. Review recent interview experiences
""",
        
        '08_interview_prep': """# Interview Preparation

Comprehensive interview preparation materials and practice problems.

## Contents
- [Practice Problems](practice/)
- [Mock Interviews](mock_interviews/)
- [Guides](guides/)
- [Checklists](checklists/)

## Quick Start
1. Review preparation guides
2. Practice coding problems
3. Do mock interviews
4. Use checklists for preparation
""",
        
        '09_curriculum': """# Structured Learning Paths

Organized curriculum for different skill levels and roles.

## Contents
- [Phase 0 - Fundamentals](phase0_fundamentals/)
- [Phase 1 - Intermediate](phase1_intermediate/)
- [Phase 2 - Advanced](phase2_advanced/)
- [Phase 3 - Expert](phase3_expert/)

## Quick Start
1. Assess your current level
2. Follow the appropriate phase
3. Complete all modules
4. Move to the next phase
""",
        
        '10_resources': """# Additional Resources

Tools, references, and external resources for interview preparation.

## Contents
- [Reference Guides](reference/)
- [Tools](tools/)
- [External Resources](external/)
- [Open Source Projects](open_source_projects.md)

## Quick Start
1. Use reference guides for quick lookup
2. Explore development tools
3. Check external resources
4. Contribute to open source projects
"""
    }
    
    for directory, template in readme_templates.items():
        readme_path = f"{directory}/README.md"
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write(template)
            print(f"Created {readme_path}")

def fix_relative_links():
    """Fix relative links in markdown files."""
    md_files = find_all_md_files()
    
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common link patterns
            # Fix links that start with ./ or ../ but should be relative
            content = re.sub(r'\]\(\.\.?/([^)]+)\)', r'](\1)', content)
            
            # Fix links that reference files without proper paths
            content = re.sub(r'\]\(([^/][^)]*\.md)\)', r'](\1)', content)
            
            # Fix links to directories that should end with /
            content = re.sub(r'\]\(([^)]+[^/])\)', r'](\1/)', content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed links in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def add_missing_resources():
    """Add missing resources and references."""
    
    # Create a comprehensive resources file
    resources_content = """# 📚 Comprehensive Resources

## 📖 Essential Books

### Computer Science Fundamentals
- **"Introduction to Algorithms"** by Cormen, Leiserson, Rivest, and Stein
- **"Computer Systems: A Programmer's Perspective"** by Bryant and O'Hallaron
- **"Operating System Concepts"** by Silberschatz, Galvin, and Gagne

### System Design
- **"Designing Data-Intensive Applications"** by Martin Kleppmann
- **"System Design Interview"** by Alex Xu
- **"High Performance Browser Networking"** by Ilya Grigorik

### Software Architecture
- **"Clean Architecture"** by Robert C. Martin
- **"Patterns of Enterprise Application Architecture"** by Martin Fowler
- **"Domain-Driven Design"** by Eric Evans

### Backend Engineering
- **"Building Microservices"** by Sam Newman
- **"Microservices Patterns"** by Chris Richardson
- **"Release It!"** by Michael Nygard

### DevOps & Infrastructure
- **"The Phoenix Project"** by Gene Kim
- **"The DevOps Handbook"** by Gene Kim
- **"Kubernetes: Up and Running"** by Kelsey Hightower

## 🛠️ Essential Tools

### Development
- **IDEs**: VS Code, IntelliJ IDEA, Vim
- **Version Control**: Git, GitHub, GitLab
- **Package Managers**: npm, yarn, pip, go mod

### Databases
- **Relational**: PostgreSQL, MySQL, SQLite
- **NoSQL**: MongoDB, Cassandra, Redis
- **Graph**: Neo4j, ArangoDB

### Infrastructure
- **Containers**: Docker, Podman
- **Orchestration**: Kubernetes, Docker Swarm
- **Cloud**: AWS, GCP, Azure

### Monitoring
- **Metrics**: Prometheus, InfluxDB
- **Visualization**: Grafana, Kibana
- **Tracing**: Jaeger, Zipkin
- **Logging**: ELK Stack, Fluentd

## 📚 Online Resources

### Courses
- **Coursera**: Computer Science Specializations
- **edX**: MIT, Stanford courses
- **Udemy**: Practical programming courses
- **Pluralsight**: Technology-specific training

### Platforms
- **LeetCode**: Coding interview practice
- **HackerRank**: Programming challenges
- **System Design Interview**: System design practice
- **Pramp**: Mock interview platform

### Documentation
- **MDN Web Docs**: Web development
- **Kubernetes Docs**: Container orchestration
- **AWS Docs**: Cloud services
- **PostgreSQL Docs**: Database reference

## 🎯 Interview Preparation

### Coding Practice
- **LeetCode**: 1000+ problems
- **HackerRank**: Algorithm challenges
- **CodeSignal**: Technical assessments
- **Pramp**: Mock coding interviews

### System Design
- **System Design Primer**: GitHub repository
- **High Scalability**: Real-world architectures
- **AWS Architecture Center**: Cloud patterns
- **Google Cloud Architecture**: Design patterns

### Behavioral
- **STAR Method**: Situation, Task, Action, Result
- **Leadership Principles**: Amazon, Google
- **Communication Skills**: Toastmasters, Coursera
- **Team Management**: Harvard Business Review

## 🔗 Useful Links

### GitHub Repositories
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [Awesome System Design](https://github.com/madd86/awesome-system-design)
- [Interview Preparation](https://github.com/jwasham/coding-interview-university)
- [Tech Interview Handbook](https://github.com/yangshun/tech-interview-handbook)

### Websites
- [System Design Interview](https://www.systemdesigninterview.com/)
- [High Scalability](http://highscalability.com/)
- [AWS Architecture Center](https://aws.amazon.com/architecture/)
- [Google Cloud Architecture](https://cloud.google.com/architecture)

### Blogs
- **Martin Fowler**: Software architecture
- **High Scalability**: System design
- **AWS Blog**: Cloud architecture
- **Google Cloud Blog**: Cloud technologies

## 📱 Mobile Apps

### Learning
- **SoloLearn**: Programming basics
- **Mimo**: Coding practice
- **Grasshopper**: JavaScript learning
- **Enki**: Daily coding challenges

### Interview Practice
- **LeetCode**: Mobile app
- **HackerRank**: Mobile challenges
- **Pramp**: Mock interviews
- **InterviewBit**: Coding practice

## 🎓 Certifications

### Cloud Platforms
- **AWS Certified Solutions Architect**
- **Google Cloud Professional Architect**
- **Microsoft Azure Solutions Architect**
- **Kubernetes Certified Administrator**

### Programming
- **Oracle Java Certification**
- **Microsoft .NET Certification**
- **Google Associate Android Developer**
- **Apple iOS Developer**

## 📺 Video Resources

### YouTube Channels
- **Gaurav Sen**: System design
- **Tech Dummies**: Microservices
- **Hussein Nasser**: Backend engineering
- **AWS**: Cloud services

### Online Courses
- **Coursera**: Computer Science
- **edX**: MIT, Stanford
- **Udemy**: Practical courses
- **Pluralsight**: Technology training

## 🏆 Practice Platforms

### Coding
- **LeetCode**: Premium features
- **HackerRank**: Skill assessments
- **CodeSignal**: Technical screening
- **Pramp**: Mock interviews

### System Design
- **System Design Interview**: Practice problems
- **High Scalability**: Case studies
- **AWS Well-Architected**: Best practices
- **Google Cloud Architecture**: Patterns

---

**Remember**: Consistent practice and real-world application are key to success in technical interviews!
"""
    
    with open('10_resources/comprehensive_resources.md', 'w') as f:
        f.write(resources_content)
    print("Created comprehensive resources file")

def create_contributing_guide():
    """Create a contributing guide."""
    contributing_content = """# Contributing to Ultimate Backend Engineering Interview Preparation

Thank you for your interest in contributing to this repository! This guide will help you get started.

## 🤝 How to Contribute

### 1. Fork the Repository
- Click the "Fork" button on the GitHub page
- Clone your fork locally
- Set up the upstream remote

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Add new content or fix existing issues
- Follow the existing structure and naming conventions
- Ensure all links work correctly
- Add proper documentation

### 4. Test Your Changes
- Verify all links work
- Check markdown formatting
- Ensure content is accurate and up-to-date

### 5. Submit a Pull Request
- Push your changes to your fork
- Create a pull request with a clear description
- Reference any related issues

## 📝 Content Guidelines

### File Structure
- Follow the existing directory structure
- Use descriptive filenames
- Include README files for new directories

### Markdown Formatting
- Use proper heading hierarchy
- Include table of contents for long files
- Add code blocks with syntax highlighting
- Use consistent formatting

### Content Quality
- Ensure accuracy and completeness
- Include practical examples
- Add code implementations where applicable
- Keep content up-to-date

## 🐛 Reporting Issues

### Bug Reports
- Use the issue template
- Provide clear steps to reproduce
- Include expected vs actual behavior
- Add relevant screenshots or logs

### Feature Requests
- Describe the feature clearly
- Explain the use case
- Consider the impact on existing content
- Provide implementation suggestions

## 📚 Content Types

### Algorithms & Data Structures
- Include Go implementations
- Add time/space complexity analysis
- Provide test cases
- Include visualizations where helpful

### System Design
- Follow the standard design process
- Include scalability considerations
- Add real-world examples
- Consider trade-offs

### Backend Engineering
- Focus on practical implementations
- Include configuration examples
- Add monitoring and debugging tips
- Consider security implications

### Interview Preparation
- Provide clear explanations
- Include practice problems
- Add mock interview scenarios
- Focus on common patterns

## 🔧 Development Setup

### Prerequisites
- Git
- Python 3.x (for scripts)
- Markdown editor
- Web browser (for testing links)

### Local Development
```bash
# Clone the repository
git clone https://github.com/your-username/nail-it.git
cd nail-it

# Create a new branch
git checkout -b feature/your-feature

# Make your changes
# Test your changes
# Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
```

## 📋 Checklist

Before submitting a pull request, ensure:

- [ ] All links work correctly
- [ ] Content is accurate and up-to-date
- [ ] Markdown formatting is correct
- [ ] Code examples are tested
- [ ] Documentation is complete
- [ ] No duplicate content
- [ ] Proper attribution for external sources

## 🎯 Areas for Contribution

### High Priority
- Fix broken links
- Update outdated information
- Add missing implementations
- Improve documentation

### Medium Priority
- Add new practice problems
- Create mock interview scenarios
- Improve existing explanations
- Add visualizations

### Low Priority
- Refactor existing content
- Add more examples
- Improve formatting
- Add more references

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Comments**: For specific feedback

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Individual file headers (where appropriate)
- Release notes

Thank you for contributing to this project!
"""
    
    with open('CONTRIBUTING.md', 'w') as f:
        f.write(contributing_content)
    print("Created contributing guide")

def create_license():
    """Create a license file."""
    license_content = """MIT License

Copyright (c) 2024 Ultimate Backend Engineering Interview Preparation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open('LICENSE', 'w') as f:
        f.write(license_content)
    print("Created license file")

def main():
    """Main function to run all fixes and additions."""
    print("🚀 Starting comprehensive link fixing and resource addition...")
    
    # Create missing README files
    print("📝 Creating missing README files...")
    create_missing_readmes()
    
    # Fix relative links
    print("🔗 Fixing relative links...")
    fix_relative_links()
    
    # Add missing resources
    print("📚 Adding missing resources...")
    add_missing_resources()
    
    # Create contributing guide
    print("📋 Creating contributing guide...")
    create_contributing_guide()
    
    # Create license
    print("⚖️ Creating license file...")
    create_license()
    
    print("✅ All fixes and additions completed!")
    print("📊 Summary:")
    print("  - Created README files for all main directories")
    print("  - Fixed relative links in markdown files")
    print("  - Added comprehensive resources")
    print("  - Created contributing guide")
    print("  - Added license file")

if __name__ == "__main__":
    main()
