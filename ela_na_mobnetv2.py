import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.models import Model
import pywt
import cv2
import PIL.Image
import PIL.ImageChops

class ForensicFeatureExtractor:
    @staticmethod
    def error_level_analysis(image_path, output_quality=90):
        """
        Perform Error Level Analysis (ELA)

        Args:
            image_path (str): Path to input image
            output_quality (int): JPEG compression quality

        Returns:
            numpy.ndarray: Scaled ELA feature map
        """
        original_image = PIL.Image.open(image_path)
        temp_compressed_path = 'temp_compressed.jpg'
        original_image.save(temp_compressed_path, 'JPEG', quality=output_quality)
        compressed_image = PIL.Image.open(temp_compressed_path)
        ela_image = PIL.ImageChops.difference(original_image, compressed_image)
        ela_array = np.array(ela_image)
        scaled_ela = ela_array * (255.0 / np.max(ela_array))
        return scaled_ela

    
    @staticmethod
    def wavelet_noise_features(image, wavelet='db4', levels=3, expected_features=9):
        """
        Extract noise features using wavelet transform.
    
        Args:
            image (numpy.ndarray): Input image
            wavelet (str): Wavelet type
            levels (int): Decomposition levels
            expected_features (int): Expected number of features for the model
    
        Returns:
            numpy.ndarray: Noise feature vector with the specified number of features
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        coeffs = pywt.wavedec2(image, wavelet, level=levels)
        noise_features = []
        for i in range(1, levels + 1):
            subbands = coeffs[i]
            for sb in subbands:
                noise_features.extend([
                    np.mean(np.abs(sb)),
                    np.std(sb),
                    np.max(np.abs(sb))
                ])
        
        # Adjust feature vector to match expected size
        if len(noise_features) > expected_features:
            noise_features = noise_features[:expected_features]  # Truncate
        elif len(noise_features) < expected_features:
            noise_features.extend([0] * (expected_features - len(noise_features)))  # Pad with zeros
        
        return np.array(noise_features)


class HybridForensicModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Create hybrid forensic detection model

        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
        """
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        image_input = Input(shape=input_shape, name='image_input')
        image_features = base_model(image_input)
        image_features = GlobalAveragePooling2D()(image_features)

        ela_input = Input(shape=(224, 224, 1), name='ela_input')
        ela_conv = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(ela_input)
        ela_pool = tf.keras.layers.MaxPooling2D((2, 2))(ela_conv)
        ela_features = GlobalAveragePooling2D()(ela_pool)

        noise_input = Input(shape=(9,), name='noise_input')

        combined_features = Concatenate()([
            image_features,
            ela_features,
            noise_input
        ])

        x = Dense(256, activation='relu')(combined_features)
        x = Dense(128, activation='relu')(x)
        output = Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs=[image_input, ela_input, noise_input], outputs=output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def prepare_input(self, image_path):
        """
        Prepare input for the model.
    
        Args:
            image_path (str): Path to input image.
    
        Returns:
            tuple: Preprocessed image, ELA map, and noise features.
        """
        # Read and preprocess the image
        image = cv2.imread(image_path)
        processed_image = cv2.resize(image, (224, 224))
        processed_image = preprocess_input(processed_image)
    
        # Generate and preprocess the ELA map
        ela_map = ForensicFeatureExtractor.error_level_analysis(image_path)
        ela_map_resized = cv2.resize(ela_map, (224, 224))
        ela_map_normalized = np.clip(ela_map_resized / 255.0, 0, 1)  # Normalize to [0, 1]
        ela_map_normalized = (ela_map_normalized * 255).astype(np.uint8)  # Convert to uint8
        
        # Convert to grayscale if needed
        if ela_map_normalized.ndim == 3 and ela_map_normalized.shape[-1] == 3:
            ela_map_normalized = cv2.cvtColor(ela_map_normalized, cv2.COLOR_BGR2GRAY)
        
        # Expand dimensions
        ela_map_expanded = np.expand_dims(ela_map_normalized, axis=-1)  # Add channel dimension
          # Add channel dimension
    
        # Extract noise features
        noise_features = ForensicFeatureExtractor.wavelet_noise_features(image)
    
        return processed_image, ela_map_expanded, noise_features


    def train(self, image_paths, labels, epochs=10, batch_size=32):
        """
        Train the hybrid forensic model

        Args:
            image_paths (list): Paths to training images
            labels (numpy.ndarray): One-hot encoded labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        X_images, X_ela, X_noise = [], [], []
        for img_path in image_paths:
            proc_img, proc_ela, proc_noise = self.prepare_input(img_path)
            X_images.append(proc_img)
            X_ela.append(proc_ela)
            X_noise.append(proc_noise)

        X_images = np.array(X_images)
        X_ela = np.array(X_ela)
        X_noise = np.array(X_noise)

        history = self.model.fit(
            [X_images, X_ela, X_noise],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        return history


def load_dataset(genuine_dir, forged_dir):
    """
    Load dataset from separate directories for genuine and forged images.

    Args:
        genuine_dir (str): Path to genuine images directory
        forged_dir (str): Path to forged images directory

    Returns:
        tuple: Image paths and labels
    """
    image_paths = []
    labels = []

    # Process genuine images
    for filename in os.listdir(genuine_dir):
        if filename.endswith(('.jpg', '.png', '.tif', '.tiff')):
            image_paths.append(os.path.join(genuine_dir, filename))
            labels.append([1, 0])  # Label for genuine images

    # Process forged images
    for filename in os.listdir(forged_dir):
        if filename.endswith(('.jpg', '.png', '.tif', '.tiff')):
            image_paths.append(os.path.join(forged_dir, filename))
            labels.append([0, 1])  # Label for forged images

    return image_paths, np.array(labels)


def main():
    genuine_image_dir = '/home/nishma/My Project/College_thesis/IFD/archive/testdata/Au'
    forged_image_dir = '/home/nishma/My Project/College_thesis/IFD/archive/testdata/Tp'

    image_paths, labels = load_dataset(genuine_image_dir, forged_image_dir)

    hybrid_model = HybridForensicModel()
    history = hybrid_model.train(image_paths, labels, epochs=5)

    # test_image_path = '/path/to/test/image.jpg'  # Replace with a test image path
    # prediction = hybrid_model.predict(test_image_path)
    # print("Forgery Probability:", prediction[0][1])

if __name__ == '__main__':
    main()
