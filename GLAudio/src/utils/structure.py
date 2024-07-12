import os

def check_and_create_folder(folder_path):
    """Check if a folder exists, and if not, create it.

    Args:
    folder_path (str): The path of the folder to check and create.

    Returns:
    None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")