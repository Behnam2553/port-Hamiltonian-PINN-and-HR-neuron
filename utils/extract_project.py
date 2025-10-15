import os
import sys
from pathlib import Path


def generate_project_map(root_dir: Path, exclude_dirs: list, exclude_files: list) -> str:
    """
    Generates a comprehensive text map of a project directory, embedding Python code.

    Args:
        root_dir: The root directory of the project (as a Path object).
        exclude_dirs: A list of directory names to exclude.
        exclude_files: A list of file names to exclude.

    Returns:
        A string containing the complete project map.
    """
    project_map = []
    project_name = root_dir.name
    project_map.append(f"# Project Directory Map: {project_name}\n")

    # Use a list to hold tuples of (depth, path) for sorting
    paths_to_process = []
    for dirpath, dirs, filenames in os.walk(root_dir, topdown=True):
        # Prune the directories to exclude
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        current_path = Path(dirpath)
        depth = len(current_path.relative_to(root_dir).parts)
        paths_to_process.append((depth, current_path, dirs, filenames))

    # Sort by depth to ensure parent directories are processed first
    paths_to_process.sort(key=lambda x: x[0])

    for depth, current_path, dirs, filenames in paths_to_process:
        relative_dir_path = current_path.relative_to(root_dir)

        # Use forward slashes for consistent, clean output
        folder_header = relative_dir_path.as_posix()
        if folder_header == ".":
            folder_header = "/"

        project_map.append(f"\n{'=' * 80}")
        project_map.append(f"## Folder: {folder_header}")
        project_map.append(f"{'-' * 60}")

        # List non-python files first
        other_files = sorted([f for f in filenames if not f.endswith('.py') and f not in exclude_files])
        if other_files:
            project_map.append("### Other Files:")
            for filename in other_files:
                project_map.append(f"- {filename}")
            project_map.append("")

        # List and embed python files
        python_files = sorted([f for f in filenames if f.endswith('.py') and f not in exclude_files])
        if python_files:
            project_map.append("### Python Source Files:")
            for filename in python_files:
                project_map.append(f"\n#### File: `{filename}`")
                project_map.append("```python")
                try:
                    with open(current_path / filename, 'r', encoding='utf-8') as file:
                        project_map.append(file.read())
                except Exception as e:
                    project_map.append(f"# Error reading file: {e}")
                project_map.append("```")

    return "\n".join(project_map)


if __name__ == "__main__":
    try:
        # --- Configuration ---
        # Dynamically determine the project's root directory.
        # This assumes the script is in 'project_root/utils/'.
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent

        # Define the output file path relative to the project root.
        output_file_name = "project_map.txt"
        output_file_path = project_root / output_file_name

        # Define directories and files to exclude from the map.
        dirs_to_exclude = ['.venv', 'venv', '.git', '__pycache__', 'results', 'data', '.idea']
        files_to_exclude = [output_file_name, 'extract_project.py', '.gitignore']

        print(f"Project root identified as: {project_root}")
        print(f"Excluding directories: {dirs_to_exclude}")
        print(f"Excluding files: {files_to_exclude}")

        # --- Execution ---
        project_map_content = generate_project_map(project_root, dirs_to_exclude, files_to_exclude)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(project_map_content)

        print(f"\n✅ Success! Project map has been saved to: {output_file_path}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
