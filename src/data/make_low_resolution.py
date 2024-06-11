import os
from PIL import Image
from pathlib import Path
import random

ROOT_DIR = Path()

input_dir = ROOT_DIR / 'data' / 'raw'
output_dir = ROOT_DIR / 'data' / 'processed'
print(ROOT_DIR)
photo_extensions = ['.jpg', '.jpeg', '.png']
for file in input_dir.iterdir():
    if file.is_file() and file.suffix.lower() in photo_extensions:
        # load an image file
        image = Image.open(file)

        
        width, height = image.size
        random_number = 0.5 # random.uniform(0.5, 0.8) ## It is no longer random
        new_width = int(width * random_number)
        new_height = int(height * random_number)
        resized_image = image.resize((new_width, new_height))

        # create output file
        output_path = output_dir / file.name
        print(output_path)

        # save the resized image to the output directory
        resized_image.save(output_path)
