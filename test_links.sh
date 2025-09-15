#!/bin/bash

# Test script to verify all links in MASTER_INDEX.md work
echo "🔍 Testing all links in MASTER_INDEX.md..."
echo "================================================"

# Extract all file paths from MASTER_INDEX.md
grep -o '`[^`]*\.md`' MASTER_INDEX.md | sed 's/`//g' | while read -r filepath; do
    if [ -f "$filepath" ]; then
        echo "✅ $filepath"
    else
        echo "❌ $filepath - FILE NOT FOUND"
    fi
done

echo ""
echo "================================================"
echo "🔍 Testing directory references..."
echo "================================================"

# Extract directory references
grep -o '`[^`]*/`' MASTER_INDEX.md | sed 's/`//g' | while read -r dirpath; do
    if [ -d "$dirpath" ]; then
        file_count=$(find "$dirpath" -name "*.md" | wc -l)
        echo "✅ $dirpath ($file_count files)"
    else
        echo "❌ $dirpath - DIRECTORY NOT FOUND"
    fi
done

echo ""
echo "================================================"
echo "📊 Summary"
echo "================================================"

# Count total files referenced
total_files=$(grep -o '`[^`]*\.md`' MASTER_INDEX.md | wc -l)
existing_files=$(grep -o '`[^`]*\.md`' MASTER_INDEX.md | sed 's/`//g' | while read -r filepath; do
    if [ -f "$filepath" ]; then echo "1"; fi
done | wc -l)

echo "Total files referenced: $total_files"
echo "Existing files: $existing_files"
echo "Missing files: $((total_files - existing_files))"

# Count total directories referenced
total_dirs=$(grep -o '`[^`]*/`' MASTER_INDEX.md | wc -l)
existing_dirs=$(grep -o '`[^`]*/`' MASTER_INDEX.md | sed 's/`//g' | while read -r dirpath; do
    if [ -d "$dirpath" ]; then echo "1"; fi
done | wc -l)

echo "Total directories referenced: $total_dirs"
echo "Existing directories: $existing_dirs"
echo "Missing directories: $((total_dirs - existing_dirs))"
