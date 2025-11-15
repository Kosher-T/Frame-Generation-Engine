import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
import os

# --- CONFIGURATION ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

def create_embedding_model():
    """
    Creates a model based on MobileNetV2 that outputs a 1D embedding vector.
    """
    # 1. Load MobileNetV2, pre-trained on ImageNet
    #    include_top=False means we DONT want the final classification layer
    base_model = MobileNetV2(weights='imagenet', 
                           include_top=False, 
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # We don't need to re-train this model.
    base_model.trainable = False

    # 2. Add a new "top" to the model
    #    GlobalAveragePooling2D takes the [7, 7, 1280] tensor and makes it [1, 1280]
    #    This is how we get our "vector" for each image!
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = preprocess_input(inputs) # Apply MobileNet's specific pre-processing
    x = base_model(x, training=False)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x) # The key step!
    
    # 3. Create the new model
    embed_model = tf.keras.Model(inputs, outputs)
    
    print("Embedding Model Created:")
    embed_model.summary()
    return embed_model

def get_image_paths(directory):
    """Recursively finds all image paths in a directory."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def generate_embeddings_from_directory(model, directory):
    """
    Generates embeddings for all images in a given directory.
    """
    print(f"\nGenerating embeddings for: {directory}")
    image_paths = get_image_paths(directory)
    
    if not image_paths:
        print(f"No images found in {directory}.")
        return None
        
    all_embeddings = []
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_imgs = []
        
        for img_path in batch_paths:
            img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = image.img_to_array(img)
            batch_imgs.append(img_array)
        
        # Predict on the whole batch
        embeddings = model.predict(np.array(batch_imgs))
        all_embeddings.extend(embeddings)
        
        print(f"  Processed {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)} images")

    return np.array(all_embeddings)

if __name__ == "__main__":
    # 1. Create the model
    embedding_model = create_embedding_model()
    
    # 2. Define your data directories
    GOOD_DATA_DIR = 'data/extracted_frames'
    BAD_DATA_DIR = 'data/drifted_frames'

    # 3. Generate the "Baseline" (Good Data)
    #    This is what we'll compare against in production
    baseline_embeddings = generate_embeddings_from_directory(embedding_model, GOOD_DATA_DIR)
    
    if baseline_embeddings is not None:
        print(f"\nSuccessfully generated {baseline_embeddings.shape[0]} baseline embeddings.")
        # Save for later use!
        np.save('baseline_embeddings.npy', baseline_embeddings)
        print("Baseline embeddings saved to 'baseline_embeddings.npy'")

    # 4. Generate "Drifted" embeddings (Bad Data)
    #    We do this now to prove our concept and find a threshold
    drifted_embeddings = generate_embeddings_from_directory(embedding_model, BAD_DATA_DIR)

    if drifted_embeddings is not None:
        print(f"\nSuccessfully generated {drifted_embeddings.shape[0]} drifted embeddings.")
        # Save for later use!
        np.save('drifted_embeddings.npy', drifted_embeddings)
        print("Drifted embeddings saved to 'drifted_embeddings.npy'")
```eof

### What this script does:

1.  **`create_embedding_model()`**: This is the core. It loads `MobileNetV2` (a powerful, pre-trained CV model) but *chops off the top layer*. It replaces it with a `GlobalAveragePooling2D` layer, which perfectly flattens the multi-dimensional tensor into a 1,280-dimension **vector**. This vector is the "embedding."
2.  **`generate_embeddings_from_directory()`**: This is a helper function that loops through all your frame folders, loads the images, processes them in batches, and uses the model to predict the embedding for each one.
3.  **`if __name__ == "__main__"`**:
    * It creates the embedding model.
    * It processes your **"Good Data"** (`data/extracted_frames`) to create `baseline_embeddings.npy`. This file is your "ground truth" of what good data looks like.
    * It processes your **"Bad Data"** (`data/drifted_frames`) to create `drifted_embeddings.npy`. We will use this file in the next step to see if we can spot the difference.

Go ahead and run this. It will take a few minutes, but at the end, you'll have the two `.npy` files that are the entire foundation for your drift detection system. Let me know when you have them!