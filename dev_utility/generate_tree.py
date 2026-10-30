from pathlib import Path

# Add any folders or files you want to hide here:
FOLDERS_TO_EXCLUDE = {
    '.venv', '.git', '.idea'
}

def build_tree_lines(dir_path: Path, prefix: str = "", lines: list = None):
    if lines is None:
        lines = []

    paths = [p for p in dir_path.iterdir() if p.name not in FOLDERS_TO_EXCLUDE]

    for i, path in enumerate(sorted(paths, key=lambda x: (x.is_file(), x.name))):
        is_last = i == len(paths) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{path.name}")

        if path.is_dir():
            next_prefix = prefix + ("    " if is_last else "│   ")
            build_tree_lines(path, next_prefix, lines)

    return lines


if __name__ == "__main__":
    target_folder = Path(__file__).parent.parent
    output_file = Path(__file__).parent / "project_structure.txt"

    if target_folder.exists() and target_folder.is_dir():
        # Compile the tree structure into a text block
        tree_content = [f"{target_folder.name}/"]
        build_tree_lines(target_folder, "", tree_content)

        # Write the text block to the output file
        output_file.write_text("\n".join(tree_content), encoding="utf-8")
        print(f"Success! Your tree diagram has been saved to: {output_file.name}")
    else:
        print(f"Error: Could not find a folder named 'Quant Model CRWD' at: {target_folder.resolve()}")
