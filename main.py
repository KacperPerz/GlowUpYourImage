import os
import cv2
import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def build_srcnn_model():
    inputs = tf.keras.Input(shape=(1024, 1024, 1))
    conv1 = tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same', name='conv1')(inputs)
    conv2 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv2')(conv1)
    outputs = tf.keras.layers.Conv2D(1, (5, 5), padding='same', name='conv3')(conv2)
    model = tf.keras.Model(inputs, outputs)
    return model

def pad_image(image, target_size=(1024, 1024)):
    h, w = image.shape[:2]
    top = (target_size[0] - h) // 2
    bottom = target_size[0] - h - top
    left = (target_size[1] - w) // 2
    right = target_size[1] - w - left
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def load_dataset(raw_dir, processed_dir, limit=None):
    high_res_images = []
    low_res_images = []
    filenames = os.listdir(raw_dir)
    if limit:
        filenames = filenames[:limit]
    for filename in filenames:
        high_res_image = cv2.imread(os.path.join(raw_dir, filename), cv2.IMREAD_COLOR)
        low_res_image = cv2.imread(os.path.join(processed_dir, filename), cv2.IMREAD_COLOR)
        
        if high_res_image is None:
            print(f"Error: Could not load high-resolution image from {os.path.join(raw_dir, filename)}")
            continue
        if low_res_image is None:
            print(f"Error: Could not load low-resolution image from {os.path.join(processed_dir, filename)}")
            continue
        
        if high_res_image.shape[:2] != (1024, 1024):
            print(f"Error: Unexpected high-resolution image dimensions for {filename}")
            continue
        
        low_res_image_padded = pad_image(low_res_image, (1024, 1024))

        high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2YCrCb)
        low_res_image_padded = cv2.cvtColor(low_res_image_padded, cv2.COLOR_BGR2YCrCb)
        
        high_res_y = high_res_image[:, :, 0]
        low_res_y = low_res_image_padded[:, :, 0]
        
        high_res_y = high_res_y.astype(np.float32) / 255.0
        low_res_y = low_res_y.astype(np.float32) / 255.0
        
        high_res_y = np.expand_dims(high_res_y, axis=-1)
        low_res_y = np.expand_dims(low_res_y, axis=-1)
        
        high_res_images.append(high_res_y)
        low_res_images.append(low_res_y)
    
    return np.array(low_res_images), np.array(high_res_images)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    padded_image = pad_image(image, (1024, 1024))
    ycrcb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycrcb)
    y = y.astype(np.float32) / 255.0
    y = np.expand_dims(y, axis=0)
    y = np.expand_dims(y, axis=-1)
    return y, cb, cr

def postprocess_image(output, cb, cr):
    output = output[0, :, :, 0]
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    ycrcb = cv2.merge((output, cb, cr))
    output_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return output_image

def super_resolve(image_path, model):
    y, cb, cr = preprocess_image(image_path)
    output = model.predict(y)
    result_image = postprocess_image(output, cb, cr)
    return result_image

def plot_bias_variance_tradeoff(history):
    """
    Function plot loss abd valitadion loss curves on one plot
    Inputs:
        history - model history
    Returns:
        None - function saves plot as png
    """

    fig = plt.figure(figsize=(10,5))
    plt.plot(range(1, EPOCHS + 1), history.history['loss'], label='Train loss')
    plt.plot(range(1, EPOCHS + 1), history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()

    plt.savefig('plots/bias_variance_tradeoff.png')
    plt.close(fig)


def show_result_and_og(input, output):
    """
    Drawing subplot with generated and original image
    Inputs: 
        input - loaded original image,
        output - generated image
    Returns:
        None - function saves plot as png
    """

    fig, (pred, true) = plt.subplots(1, 2, figsize=(20, 10))

    # Plotting the generated imaage
    pred.imshow(output)
    pred.axis('off')
    pred.set_title('Generated Image', fontsize=22)

    # Plotting the original image
    true.imshow(input)
    true.axis('off')
    true.set_title('Original Image', fontsize=22)
    plt.tight_layout()

    plt.savefig('plots/result.png')
    plt.close(fig)

    


if __name__ == "__main__":

    RAW_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    EPOCHS = 50
    BATCH_SIZE = 10

    low_res_images, high_res_images = load_dataset(RAW_DIR, PROCESSED_DIR, limit=20)

    with mlflow.start_run():

        model = build_srcnn_model()
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')

        history = model.fit(low_res_images, high_res_images, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

        # Save hiperparameters
        mlflow.log_param('epochs', EPOCHS)
        mlflow.log_param('optimizer', 'adam')
        mlflow.log_param('batch_size', BATCH_SIZE)

        # Save metrics
        mlflow.log_metric('final_loss', history.history['loss'][-1])
        mlflow.log_metric('final_val_loss', history.history['val_loss'][-1])

        # Save plots
        input_image_path = 'data/processed/seed0208.png'
        input_image = plt.imread(input_image_path)
        output_image = super_resolve(input_image_path, model)

        plot_bias_variance_tradeoff(history)
        show_result_and_og(input_image, output_image)

        mlflow.log_artifact('plots/bias_variance_tradeoff.png')
        mlflow.log_artifact('plots/result.png')
    
    # Log model
    mlflow.tensorflow.log_model(model, 'srcnn_model')

    # MLFlow zapisze i spickluje model
    #model.save('srcnn_model.h5')

    

    cv2.imwrite('high_resolution_image.jpg', output_image)
