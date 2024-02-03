import os


def find_project_root() -> str:
    """
    Finds and returns the root path of the project.

    Returns:
        str: The absolute path to the project root.
    """
    try:
        # Get the absolute path of the current script/module
        current_script_path = os.path.abspath(__file__)

        # Go up one directory level to get the project root path
        project_root_path = os.path.dirname(os.path.dirname(current_script_path))

        return project_root_path
    except Exception as e:
        print(f"Error in find_project_root: {e}")
        return None
