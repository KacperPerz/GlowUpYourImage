import os
import numpy as np
import cv2
import tensorflow as tf

def build_srcnn_model():
    inputs = tf.keras.Input(shape=(1024, 1024, 1))  # Fixing the input size
    conv1 = tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same', name='conv1')(inputs)
    conv2 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv2')(conv1)
    outputs = tf.keras.layers.Conv2D(1, (5, 5), padding='same', name='conv3')(conv2)
    model = tf.keras.Model(inputs, outputs)
    return model

def resize_image(image, target_size=(1024, 1024)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

def load_image_pairs(raw_dir, processed_dir, filenames):
    for filename in filenames:
        high_res_image = cv2.imread(os.path.join(raw_dir, filename), cv2.IMREAD_COLOR)
        low_res_image = cv2.imread(os.path.join(processed_dir, filename), cv2.IMREAD_COLOR)

        if high_res_image is None or low_res_image is None:
            continue
        
        # Ensure high-resolution images are 1024x1024
        if high_res_image.shape[:2] != (1024, 1024):
            continue
        
        # Resize low-res images to match the high-res images
        low_res_image_resized = resize_image(low_res_image, (1024, 1024))

        high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2YCrCb)
        low_res_image_resized = cv2.cvtColor(low_res_image_resized, cv2.COLOR_BGR2YCrCb)
        
        high_res_y = high_res_image[:, :, 0]
        low_res_y = low_res_image_resized[:, :, 0]
        
        high_res_y = high_res_y.astype(np.float32) / 255.0
        low_res_y = low_res_y.astype(np.float32) / 255.0
        
        high_res_y = np.expand_dims(high_res_y, axis=-1)
        low_res_y = np.expand_dims(low_res_y, axis=-1)
        
        yield low_res_y, high_res_y

def data_generator(raw_dir, processed_dir, batch_size):
    filenames = os.listdir(raw_dir)
    while True:
        np.random.shuffle(filenames)
        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i:i + batch_size]
            batch_low_res, batch_high_res = zip(*load_image_pairs(raw_dir, processed_dir, batch_filenames))
            yield np.array(batch_low_res), np.array(batch_high_res)

# Function to preprocess the image for inference
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    resized_image = resize_image(image, (1024, 1024))
    ycrcb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycrcb)
    y = y.astype(np.float32) / 255.0
    y = np.expand_dims(y, axis=0)
    y = np.expand_dims(y, axis=-1)
    return y, cb, cr

# Function to post-process the output image
def postprocess_image(output, cb, cr):
    output = output[0, :, :, 0]
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    ycrcb = cv2.merge((output, cb, cr))
    output_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return output_image

# Super-resolution function
def super_resolve(image_path, model):
    y, cb, cr = preprocess_image(image_path)
    output = model.predict(y)
    result_image = postprocess_image(output, cb, cr)
    return result_image

if __name__ == "__main__":
    model = build_srcnn_model()
    model.compile(optimizer='adam', loss='mean_squared_error')

    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    batch_size = 8

    train_gen = data_generator(raw_dir, processed_dir, batch_size)
    steps_per_epoch = len(os.listdir(raw_dir)) // batch_size

    model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=100, validation_steps=steps_per_epoch * 0.1)

    model.save('srcnn_model.h5')

    input_image_path = 'processed/your_low_res_image.jpg'
    output_image = super_resolve(input_image_path, model)

    cv2.imwrite('high_resolution_image.jpg', output_image)
