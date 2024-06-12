import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

def main(image_path, output_path):
    model = load_model()
    output_image = super_resolve(image_path, model)
    cv2.imwrite(output_path, output_image)

    # Display images
    original_image = cv2.imread(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Upscaled Image")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    input_image_path = 'raw.png'  # Replace with your input image path
    output_image_path = 'upscaled_image.jpg'  # Output image path
    main(input_image_path, output_image_path)
