#!/usr/bin/env python3
"""
Confluence Publisher Script for Master Engineer Curriculum
This script helps prepare and format content for Confluence Smart Publisher
"""

import os
import re
import json
from pathlib import Path

class ConfluencePublisher:
    def __init__(self, curriculum_path="09_curriculum"):
        self.curriculum_path = Path(curriculum_path)
        self.confluence_content = []
        
    def load_curriculum_structure(self):
        """Load the curriculum structure and create Confluence-ready content"""
        structure = {
            "phase0": {
                "name": "Phase 0: Fundamentals",
                "modules": ["mathematics", "programming", "cs-basics", "software-engineering"],
                "status": "Complete"
            },
            "phase1": {
                "name": "Phase 1: Intermediate", 
                "modules": ["advanced-dsa", "os-deep-dive", "database-systems", 
                           "web-development", "api-design", "system-design-basics"],
                "status": "Complete"
            },
            "phase2": {
                "name": "Phase 2: Advanced",
                "modules": ["advanced-algorithms", "cloud-architecture", "machine-learning",
                           "performance-engineering", "security-engineering", "distributed-systems"],
                "status": "Complete"
            },
            "phase3": {
                "name": "Phase 3: Expert",
                "modules": ["technical-leadership", "architecture-design", "innovation-research",
                           "mentoring-coaching", "strategic-planning"],
                "status": "Complete"
            }
        }
        return structure
    
    def generate_confluence_macros(self):
        """Generate Confluence-specific macros for the curriculum"""
        macros = {
            "toc": "{toc:printable=true|style=square|maxLevel=3|indent=20px|minLevel=1|exclude=[1//2]|type=list|outline=clear|include=.*}",
            "info": "{info:title=Master Engineer Curriculum}\nA comprehensive learning path from fundamentals to distinguished engineer level with 550+ code examples and 100+ visual diagrams.\n{info}",
            "warning": "{warning:title=Learning Path}\nStart with Phase 0 fundamentals before progressing to advanced topics. Each phase builds upon the previous one.\n{warning}",
            "note": "{note:title=Prerequisites}\n- Basic programming knowledge\n- Access to development environment\n- Git for version control\n{note}",
            "status_complete": "{status:colour=Green|title=Complete}",
            "card_template": "{card:title={title}|icon={icon}|bgColor={color}}\n{content}\n{card}",
            "table_template": "{table:title={title}|sortable=true}\n{content}\n{table}",
            "code_template": "{code:language={language}|title={title}|linenumbers=true|collapse=false}\n{content}\n{code}",
            "panel_template": "{panel:title={title}|borderStyle=solid|borderColor=#ccc|titleBGColor={titleColor}|bgColor={bgColor}}\n{content}\n{panel}"
        }
        return macros
    
    def create_phase_cards(self, structure):
        """Create Confluence cards for each phase"""
        phase_cards = []
        colors = {
            "phase0": "#e8f5e8",
            "phase1": "#fff3cd", 
            "phase2": "#d1ecf1",
            "phase3": "#f8d7da"
        }
        icons = {
            "phase0": "book",
            "phase1": "rocket",
            "phase2": "star", 
            "phase3": "crown"
        }
        
        for phase_id, phase_data in structure.items():
            duration = {
                "phase0": "8-12 weeks",
                "phase1": "12-16 weeks",
                "phase2": "16-20 weeks", 
                "phase3": "20-24 weeks"
            }[phase_id]
            
            focus = {
                "phase0": "Building strong technical foundations",
                "phase1": "Developing intermediate skills and system design knowledge",
                "phase2": "Mastering advanced concepts and distributed systems",
                "phase3": "Leadership, architecture, and strategic thinking"
            }[phase_id]
            
            content = f"**Duration**: {duration}\n**Focus**: {focus}\n**Prerequisites**: {phase_id.replace('phase', 'Phase ')} completion\n**Modules**: {len(phase_data['modules'])}"
            
            card = f"{{card:title={phase_data['name']}|icon={icons[phase_id]}|bgColor={colors[phase_id]}}}\n{content}\n{{card}}"
            phase_cards.append(card)
            
        return phase_cards
    
    def create_statistics_table(self):
        """Create the curriculum statistics table"""
        table_content = """| Phase | Modules | Implementation Files | Code Examples | Mermaid Diagrams |
| Phase 0 | 4 | 8 | 120+ | 20+ |
| Phase 1 | 6 | 18 | 200+ | 40+ |
| Phase 2 | 6 | 10 | 150+ | 30+ |
| Phase 3 | 5 | 5 | 80+ | 15+ |
| **Total** | **21** | **41** | **550+** | **105+** |"""
        
        return f"{{table:title=Curriculum Statistics|sortable=true}}\n{table_content}\n{{table}}"
    
    def create_implementation_status_panel(self):
        """Create the implementation status panel"""
        content = """✅ 35+ comprehensive implementation files
✅ 500+ production-ready code examples
✅ 100+ Mermaid diagrams
✅ Complete coverage of all phases
✅ Cross-linked learning paths"""
        
        return f"{{panel:title=Implementation Status|borderStyle=solid|borderColor=#ccc|titleBGColor=#f7d6c1|bgColor=#ffffce}}\n{content}\n{{panel}}"
    
    def generate_confluence_content(self):
        """Generate the complete Confluence-ready content"""
        structure = self.load_curriculum_structure()
        macros = self.generate_confluence_macros()
        
        content = []
        
        # Header
        content.append("# Master Engineer Curriculum - Ready for Confluence")
        content.append("*Copy and paste this content directly into Confluence Smart Publisher*")
        content.append("")
        content.append("---")
        content.append("")
        
        # Table of Contents
        content.append(macros["toc"])
        content.append("")
        
        # Info, Warning, Note panels
        content.append(macros["info"])
        content.append("")
        content.append(macros["warning"])
        content.append("")
        content.append(macros["note"])
        content.append("")
        content.append("---")
        content.append("")
        
        # Curriculum Overview
        content.append("## 🎯 Curriculum Overview")
        content.append("")
        content.append("The Master Engineer Curriculum is a complete, step-by-step learning path designed to take engineers from absolute basics to distinguished 20-year senior/Staff/Distinguished Engineer level. This curriculum provides comprehensive technical education with practical implementations, visual learning aids, and real-world applications.")
        content.append("")
        content.append("---")
        content.append("")
        
        # Statistics Table
        content.append("## 📊 Curriculum Statistics")
        content.append("")
        content.append(self.create_statistics_table())
        content.append("")
        content.append("---")
        content.append("")
        
        # Phase Cards
        phase_cards = self.create_phase_cards(structure)
        for i, (phase_id, phase_data) in enumerate(structure.items()):
            content.append(f"## 📚 {phase_data['name']}")
            content.append("")
            content.append(f"{{status:colour=Green|title=Complete}}")
            content.append(f"{phase_data['name']} - Complete")
            content.append("{status}")
            content.append("")
            content.append(phase_cards[i])
            content.append("")
            
            # Add module links (simplified for this example)
            content.append("### Modules")
            for module in phase_data['modules']:
                module_name = module.replace('-', ' ').title()
                content.append(f"- [{module_name}](./{phase_id.replace('phase', 'phase')}_{phase_id.split('phase')[1]}_intermediate/{module}/README.md)")
            content.append("")
            content.append("---")
            content.append("")
        
        # Key Features
        content.append("## 🎯 Key Features")
        content.append("")
        content.append(self.create_implementation_status_panel())
        content.append("")
        
        # Add more sections...
        content.append("### ✅ Comprehensive Coverage")
        content.append("- Complete learning path from beginner to expert")
        content.append("- All major engineering concepts covered")
        content.append("- Industry-relevant content and best practices")
        content.append("")
        
        content.append("### ✅ Practical Implementation")
        content.append("- Production-ready Golang and Node.js code")
        content.append("- Real-world examples and use cases")
        content.append("- Extensive testing and error handling")
        content.append("")
        
        content.append("### ✅ Visual Learning")
        content.append("- 100+ Mermaid diagrams for complex concepts")
        content.append("- Process flows and architecture diagrams")
        content.append("- Algorithm visualizations")
        content.append("")
        
        content.append("### ✅ Cross-Linked Content")
        content.append("- Interconnected learning paths")
        content.append("- Cross-references between modules")
        content.append("- Comprehensive navigation system")
        content.append("")
        
        content.append("---")
        content.append("")
        
        # Footer
        content.append("## 📞 Support")
        content.append("")
        content.append("For questions, feedback, or contributions to the Master Engineer Curriculum, please refer to the individual module documentation or contact the curriculum maintainers.")
        content.append("")
        content.append("---")
        content.append("")
        content.append("{quote}")
        content.append("The Master Engineer Curriculum provides everything needed for engineers to progress from fundamentals to distinguished engineer level with practical, hands-on learning experiences.")
        content.append("{quote}")
        content.append("")
        content.append("---")
        content.append("")
        content.append("*Last Updated: December 2024*")
        content.append("*Version: 1.0.0*")
        content.append("*Status: Complete*")
        
        return "\n".join(content)
    
    def save_confluence_content(self, output_file="CONFLUENCE_READY_CONTENT.md"):
        """Save the Confluence-ready content to a file"""
        content = self.generate_confluence_content()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Confluence content generated and saved to {output_file}")
        print("📋 Ready to copy-paste into Confluence Smart Publisher!")
        
        return output_file

def main():
    """Main function to generate Confluence content"""
    print("🚀 Starting Confluence Publisher for Master Engineer Curriculum...")
    
    publisher = ConfluencePublisher()
    output_file = publisher.save_confluence_content()
    
    print(f"\n📚 Content ready for Confluence!")
    print(f"📁 File: {output_file}")
    print("\n📋 Next steps:")
    print("1. Open Confluence Smart Publisher")
    print("2. Copy content from the generated file")
    print("3. Paste into your engineering page")
    print("4. Publish and share with your team!")

if __name__ == "__main__":
    main()
