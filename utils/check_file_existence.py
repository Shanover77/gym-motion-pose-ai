import os


def check_file_existence(file_path):
    """
    Check if a file exists at the given file path.

    Parameters:
    - file_path (str): The path of the file to check.

    Returns:
    - bool: True if the file exists, False otherwise.
    """
    try:
        if os.path.exists(file_path):
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while checking file existence: {e}")
        return False
