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
        # Open the original image
        original_image = PIL.Image.open(image_path)
        
        # Create a temporary file to save the re-compressed image
        temp_compressed_path = 'temp_compressed.jpg'
        original_image.save(temp_compressed_path, 'JPEG', quality=output_quality)
        
        # Reload the compressed image
        compressed_image = PIL.Image.open(temp_compressed_path)
        
        # Calculate the difference between original and compressed images
        ela_image = PIL.ImageChops.difference(original_image, compressed_image)
        
        # Scale the difference to enhance visibility
        ela_array = np.array(ela_image)
        scaled_ela = ela_array * (255.0 / np.max(ela_array))
        
        return scaled_ela
    
    @staticmethod
    def wavelet_noise_features(image, wavelet='db4', levels=3):
        """
        Extract noise features using wavelet transform
        
        Args:
            image (numpy.ndarray): Input image
            wavelet (str): Wavelet type
            levels (int): Decomposition levels
        
        Returns:
            numpy.ndarray: Noise feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Wavelet decomposition
        coeffs = pywt.wavedec2(image, wavelet, level=levels)
        
        # Extract noise features
        noise_features = []
        for i in range(1, levels + 1):
            # High-frequency subbands (details)
            subbands = coeffs[i]
            
            # Compute features for each subband
            for sb in subbands:
                noise_features.extend([
                    np.mean(np.abs(sb)),  # Mean absolute detail
                    np.std(sb),           # Standard deviation
                    np.max(np.abs(sb))    # Max absolute value
        ])
        
        return np.array(noise_features)

class HybridForensicModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Create hybrid forensic detection model
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
        """
        # Base MobileNetV2 model for image feature extraction
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Image input branch
        image_input = Input(shape=input_shape, name='image_input')
        image_features = base_model(image_input)
        image_features = GlobalAveragePooling2D()(image_features)
        
        # ELA input branch
        ela_input = Input(shape=(224, 224, 1), name='ela_input')
        ela_conv = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(ela_input)
        ela_pool = tf.keras.layers.MaxPooling2D((2, 2))(ela_conv)
        ela_features = GlobalAveragePooling2D()(ela_pool)
        
        # Noise features input branch
        noise_input = Input(shape=(9,), name='noise_input')
        
        # Combine features
        combined_features = Concatenate()([
            image_features, 
            ela_features, 
            noise_input
        ])
        
        # Additional dense layers
        x = Dense(256, activation='relu')(combined_features)
        x = Dense(128, activation='relu')(x)
        
        # Output layer
        output = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(
            inputs=[image_input, ela_input, noise_input], 
            outputs=output
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_input(self, image_path):
        """
        Prepare input for the model
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            tuple: Preprocessed image, ELA map, and noise features
        """
        # Read image
        image = cv2.imread(image_path)
        
        # Preprocess image for MobileNetV2
        processed_image = cv2.resize(image, (224, 224))
        processed_image = preprocess_input(processed_image)
        
        # Generate ELA map
        ela_map = ForensicFeatureExtractor.error_level_analysis(image_path)
        ela_map_resized = cv2.resize(ela_map, (224, 224))
        ela_map_normalized = ela_map_resized / 255.0
        ela_map_expanded = np.expand_dims(ela_map_normalized, axis=-1)
        
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
        # Prepare training data
        X_images = []
        X_ela = []
        X_noise = []
        
        for img_path in image_paths:
            proc_img, proc_ela, proc_noise = self.prepare_input(img_path)
            X_images.append(proc_img)
            X_ela.append(proc_ela)
            X_noise.append(proc_noise)
        
        # Convert to numpy arrays
        X_images = np.array(X_images)
        X_ela = np.array(X_ela)
        X_noise = np.array(X_noise)
        
        # Train model
        history = self.model.fit(
            [X_images, X_ela, X_noise], 
            labels, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=0.2
        )
        
        return history
    
    def predict(self, image_path):
        """
        Predict forgery likelihood
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        # Prepare input
        processed_image, ela_map, noise_features = self.prepare_input(image_path)
        
        # Predict
        prediction = self.model.predict([
            np.expand_dims(processed_image, axis=0),
            np.expand_dims(ela_map, axis=0),
            np.expand_dims(noise_features, axis=0)
        ])
        
        return prediction

# Example usage (pseudo-code)
def main():
    # Prepare your dataset
    # image_paths: list of image file paths
    # labels: one-hot encoded labels
    
    # Initialize hybrid model
    hybrid_model = HybridForensicModel()
    
    # Train model
    history = hybrid_model.train(
        image_paths, 
        labels, 
        epochs=10
    )
    
    # Predict on new image
    test_image_path = 'suspicious_image.jpg'
    prediction = hybrid_model.predict(test_image_path)
    print("Forgery Probability:", prediction[0][1])

# Note: Actual implementation requires prepared dataset

main()