#!/bin/bash

echo "# 🔍 Internal Link Validation Report"
echo ""
echo "> **Comprehensive scan of all internal links in the repository**"
echo ""
echo "## 📊 Summary"
echo ""

# Initialize counters
total_links=0
working_links=0
broken_links=0
files_with_links=0

# Create temporary files
temp_links=$(mktemp)
temp_results=$(mktemp)

echo "| File | Line | Link Text | Link Target | Status |"
echo "|------|------|-----------|-------------|--------|"

# Find all markdown files and process them
find . -name "*.md" -type f | while read -r file; do
    # Get the directory of the current file
    file_dir=$(dirname "$file")
    
    # Extract all markdown links from the file
    grep -n "\[.*\](" "$file" | while IFS=: read -r line_num line_content; do
        # Extract link text and URL using sed
        link_text=$(echo "$line_content" | sed -n 's/.*\[\([^]]*\)\]([^)]*).*/\1/p')
        link_url=$(echo "$line_content" | sed -n 's/.*\[[^]]*\](\([^)]*\)).*/\1/p')
        
        # Skip if no link found
        if [ -z "$link_url" ]; then
            continue
        fi
        
        # Skip external links (http/https)
        if [[ "$link_url" =~ ^https?:// ]]; then
            continue
        fi
        
        # Skip anchor links (starting with #)
        if [[ "$link_url" =~ ^# ]]; then
            continue
        fi
        
        # Increment total links counter
        total_links=$((total_links + 1))
        
        # Resolve the target path relative to the current file's directory
        if [[ "$link_url" =~ ^\.\./ ]]; then
            # Relative path going up directories
            target_path="$file_dir/$link_url"
        elif [[ "$link_url" =~ ^\./ ]]; then
            # Relative path in same directory
            target_path="$file_dir/$link_url"
        elif [[ "$link_url" =~ ^/ ]]; then
            # Absolute path from repo root
            target_path=".$link_url"
        else
            # Relative path in same directory
            target_path="$file_dir/$link_url"
        fi
        
        # Clean up the path
        target_path=$(echo "$target_path" | sed 's|//|/|g')
        
        # Check if the target file exists
        if [ -f "$target_path" ]; then
            status="✅ Working"
            working_links=$((working_links + 1))
        else
            status="❌ Broken"
            broken_links=$((broken_links + 1))
        fi
        
        # Escape special characters for markdown table
        link_text_escaped=$(echo "$link_text" | sed 's/|/\\|/g')
        link_url_escaped=$(echo "$link_url" | sed 's/|/\\|/g')
        file_escaped=$(echo "$file" | sed 's/|/\\|/g')
        
        # Output the result
        echo "| $file_escaped | $line_num | $link_text_escaped | $link_url_escaped | $status |"
    done
done

echo ""
echo "## 📈 Statistics"
echo ""
echo "- **Total Internal Links Checked**: $total_links"
echo "- **Working Links**: $working_links"
echo "- **Broken Links**: $broken_links"
echo "- **Success Rate**: $(( working_links * 100 / total_links ))%"

# Clean up
rm -f "$temp_links" "$temp_results"
