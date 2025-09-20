import os
import re
from pathlib import Path

def generate_toc(root_dir):
    toc = {}
    root_path = Path(root_dir)
    for path in sorted(root_path.rglob('*.md')):
        relative_path = path.relative_to(root_path)
        parts = list(relative_path.parts)
        
        # Create a title from the filename
        title = parts[-1].replace('.md', '').replace('_', ' ').replace('-', ' ').title()
        
        # Navigate the TOC dictionary
        current_level = toc
        for part in parts[:-1]:
            part_title = part.replace('_', ' ').replace('-', ' ').title()
            if part_title not in current_level:
                current_level[part_title] = {}
            current_level = current_level[part_title]
        
        current_level[title] = str(relative_path)
        
    return toc

def format_toc_markdown(toc, level=1):
    lines = []
    for title, value in toc.items():
        if isinstance(value, dict):
            lines.append(f"{'#' * level} {title}")
            lines.extend(format_toc_markdown(value, level + 1))
        else:
            lines.append(f"- [{title}]({value})")
    return lines

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    toc_data = generate_toc(repo_root)
    
    # Update MASTER_INDEX.md
    master_index_path = repo_root / 'MASTER_INDEX.md'
    if master_index_path.exists():
        with open(master_index_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            
            # Find a marker to replace or append
            toc_marker = "## 📚 Full Repository Index"
            if toc_marker in content:
                # More robust replacement logic might be needed
                pass
            else:
                f.write(f"\n\n{toc_marker}\n\n")
                f.write("\n".join(format_toc_markdown(toc_data)))

    # Update READMEs in each main directory
    for main_dir in repo_root.iterdir():
        if main_dir.is_dir() and main_dir.name.startswith('0'):
            readme_path = main_dir / 'README.md'
            dir_toc = generate_toc(main_dir)
            
            readme_content = f"# {main_dir.name.replace('_', ' ').title()}\n\n"
            readme_content += "\n".join(format_toc_markdown(dir_toc, level=2))
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

    print("TOCs updated.")
