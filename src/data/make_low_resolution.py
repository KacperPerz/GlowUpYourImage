import os
from PIL import Image
from pathlib import Path
import random

ROOT_DIR = Path().resolve().parent

input_dir = ROOT_DIR / 'data' / 'raw'
output_dir = ROOT_DIR / 'data' / 'processed'

photo_extensions = ['.jpg', '.jpeg', '.png']
for file in input_dir.iterdir():
    if file.is_file() and file.suffix.lower() in photo_extensions:
        # load an image file
        image = Image.open(file)

        # reduce the size of the image by random number between 0.5 and 0.8
        width, height = image.size
        random_number = random.uniform(0.5, 0.8)
        new_width = int(width * random_number)
        new_height = int(height * random_number)
        resized_image = image.resize((new_width, new_height))

        # create output file
        output_path = output_dir / file.name
        print(output_path)

        # save the resized image to the output directory
        resized_image.save(output_path)
