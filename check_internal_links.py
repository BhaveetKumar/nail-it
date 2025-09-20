#!/usr/bin/env python3
"""
Internal Link Checker for Markdown Files
Scans all .md files and checks if internal links point to existing files.
"""

import os
import re
import sys
from pathlib import Path

def find_markdown_files(root_dir):
    """Find all markdown files in the repository."""
    md_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

def extract_links_from_file(file_path):
    """Extract all markdown links from a file."""
    links = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Find all markdown links [text](url)
                matches = re.finditer(r'\[([^\]]*)\]\(([^)]+)\)', line)
                for match in matches:
                    link_text = match.group(1)
                    link_url = match.group(2)
                    links.append({
                        'file': file_path,
                        'line': line_num,
                        'text': link_text,
                        'url': link_url
                    })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return links

def is_external_link(url):
    """Check if a link is external (http/https)."""
    return url.startswith(('http://', 'https://'))

def is_anchor_link(url):
    """Check if a link is an anchor link (starts with #)."""
    return url.startswith('#')

def resolve_link_path(file_path, link_url):
    """Resolve the absolute path of a link relative to the file."""
    file_dir = os.path.dirname(file_path)
    
    if link_url.startswith('/'):
        # Absolute path from repo root
        return os.path.join(os.getcwd(), link_url[1:])
    elif link_url.startswith('../'):
        # Relative path going up directories
        return os.path.normpath(os.path.join(file_dir, link_url))
    elif link_url.startswith('./'):
        # Relative path in same directory
        return os.path.normpath(os.path.join(file_dir, link_url))
    else:
        # Relative path in same directory
        return os.path.normpath(os.path.join(file_dir, link_url))

def check_link_exists(resolved_path):
    """Check if the resolved path exists."""
    return os.path.exists(resolved_path)

def main():
    """Main function to check all internal links."""
    root_dir = os.getcwd()
    
    print("# 🔍 Internal Link Validation Report")
    print("")
    print("> **Comprehensive scan of all internal links in the repository**")
    print("")
    print("## 📊 Results")
    print("")
    print("| File | Line | Link Text | Link Target | Status |")
    print("|------|------|-----------|-------------|--------|")
    
    total_links = 0
    working_links = 0
    broken_links = 0
    
    # Find all markdown files
    md_files = find_markdown_files(root_dir)
    
    for file_path in md_files:
        # Extract links from file
        links = extract_links_from_file(file_path)
        
        for link in links:
            url = link['url']
            
            # Skip external links
            if is_external_link(url):
                continue
                
            # Skip anchor links
            if is_anchor_link(url):
                continue
            
            total_links += 1
            
            # Resolve the link path
            resolved_path = resolve_link_path(file_path, url)
            
            # Check if the target exists
            if check_link_exists(resolved_path):
                status = "✅ Working"
                working_links += 1
            else:
                status = "❌ Broken"
                broken_links += 1
            
            # Escape special characters for markdown table
            file_escaped = link['file'].replace('|', '\\|')
            text_escaped = link['text'].replace('|', '\\|')
            url_escaped = url.replace('|', '\\|')
            
            print(f"| {file_escaped} | {link['line']} | {text_escaped} | {url_escaped} | {status} |")
    
    print("")
    print("## 📈 Statistics")
    print("")
    print(f"- **Total Internal Links Checked**: {total_links}")
    print(f"- **Working Links**: {working_links}")
    print(f"- **Broken Links**: {broken_links}")
    if total_links > 0:
        success_rate = (working_links * 100) // total_links
        print(f"- **Success Rate**: {success_rate}%")
    else:
        print("- **Success Rate**: N/A (no internal links found)")

if __name__ == "__main__":
    main()
