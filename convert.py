#!/usr/bin/env python3
"""
convert_and_resize.py

A script to:
1. Convert WebP images to PNG (removing the original WebP).
2. Resize images so both width and height are <= 800px.
3. Log all processing steps to the console.

Requires: Pillow (pip install Pillow)
"""

import os
from PIL import Image

def convert_to_png(source_path: str) -> str:
    """
    Converts a WebP image to PNG and removes the original file.
    Returns the newly created PNG's filepath.
    """
    directory, filename = os.path.split(source_path)
    base, _ = os.path.splitext(filename)
    new_filename = base + ".png"
    new_filepath = os.path.join(directory, new_filename)

    with Image.open(source_path) as im:
        im.save(new_filepath, "PNG")
    os.remove(source_path)

    print(f"Converted {filename} to {new_filename} and removed the original.")
    return new_filepath

def resize_image(image_path: str, max_size: int = 800):
    """
    Resizes the image to ensure neither dimension exceeds max_size.
    Overwrites the existing file with the resized image.
    """
    with Image.open(image_path) as im:
        width, height = im.size
        if width > max_size or height > max_size:
            # Calculate new dimensions, preserving aspect ratio.
            if width > height:
                ratio = max_size / float(width)
            else:
                ratio = max_size / float(height)

            new_width = int(width * ratio)
            new_height = int(height * ratio)

            # Use a high-quality resampling filter.
            resized_im = im.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_im.save(image_path)
            print(f"Resized {os.path.basename(image_path)} to {new_width}x{new_height}.")
        else:
            print(f"No resize needed for {os.path.basename(image_path)}, size is {width}x{height}.")

def main():
    # Directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Extensions we want to handle
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")

    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(current_dir):
        for filename in files:
            # Check if the file has a valid extension
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(root, filename)

                # Convert .webp to .png (if needed)
                if filename.lower().endswith(".webp"):
                    file_path = convert_to_png(file_path)

                # Resize the image if it's larger than 800px in any dimension
                resize_image(file_path)

if __name__ == "__main__":
    main()
