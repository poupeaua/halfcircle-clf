"""
Utils code for data module
"""

import os
import requests

def get_image_paths(directory, extensions={".jpg", ".jpeg", ".png"}):
    """
    Recursively retrieves a list of image file paths from a given directory.
    
    :param directory: Root directory to search for images.
    :param extensions: Set of valid image file extensions.
    :return: List of image file paths.
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        files: list[str]
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def download_image(image_url, save_path):
    # Send a GET request to the Picsum API
    response = requests.get(image_url, stream=True)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the image file in write-binary mode and save it to disk
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    else:
        print("Failed to retrieve image")
        pass