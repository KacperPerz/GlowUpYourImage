# upscale.py

import os
import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path='srcnn_model.h5'):
    return tf.keras.models.load_model(model_path)

def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

def preprocess_image(image_path, target_size=(337, 337)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    resized_image = resize_image(image, target_size)
    ycrcb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycrcb)
    y = y.astype(np.float32) / 255.0
    y = np.expand_dims(y, axis=0)
    y = np.expand_dims(y, axis=-1)
    return y, cb, cr

def postprocess_image(output, cb, cr):
    output = output[0, :, :, 0]
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    cb = resize_image(cb, (output.shape[0], output.shape[1]))
    cr = resize_image(cr, (output.shape[0], output.shape[1]))
    ycrcb = cv2.merge((output, cb, cr))
    output_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return output_image

def super_resolve(image_path, model):
    y, cb, cr = preprocess_image(image_path, target_size=(337, 337))
    y_upscaled = resize_image(y[0], (1024, 1024))
    y_upscaled = np.expand_dims(y_upscaled, axis=0)
    output = model.predict(y_upscaled)
    result_image = postprocess_image(output, cb, cr)
    return result_image

def process_image(image_path, output_path):
    model = load_model()

    # Load original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")

    # Resize original image to 1024x1024
    original_resized = resize_image(original_image, (1024, 1024))

    # Downscale original image to 337x337
    downscaled_image = resize_image(original_image, (337, 337))

    # Upscale using the model
    upscaled_image = super_resolve(image_path, model)

    # Create a new image to hold the original, downscaled, and upscaled images side by side
    combined_image = np.hstack((original_resized, resize_image(downscaled_image, (1024, 1024)), upscaled_image))

    # Save the combined image
    cv2.imwrite(output_path, upscaled_image)

    return combined_image
