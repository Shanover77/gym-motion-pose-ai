import os


def ensure_folder_exists(folder_path):
    """
    Ensure that the specified folder exists. If not, create it.

    Args:
        folder_path (str): Path of the folder to be ensured.

    Returns:
        bool: True if the folder exists or is created successfully, False otherwise.
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return True
    except Exception as e:
        print(f"Error ensuring folder exists: {e}")
        return False
