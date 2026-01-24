"""
Load and process the produce dataset.
Loading the images/train directory and associated label files.
This module will read the dataset structure and prepare it for training.
"""
import os
import yaml
import cv2
import numpy as np

from glob import glob
from pathlib import Path


# ============================================================
# Utility Functions
# ============================================================

def load_yaml_classes(yaml_path):
    """
    Load class names and metadata from YOLO data.yaml file.

    Parameters
    ----------
    yaml_path : str
        Path to the data.yaml file.

    Returns
    -------
    dict
        Parsed YAML content including class names and paths.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def yolo_to_pixel_coords(box, img_w, img_h):
    """
    Convert YOLO normalized bounding box to pixel coordinates.

    YOLO format: (class_id, x_center, y_center, width, height)

    Parameters
    ----------
    box : list[float]
        YOLO bounding box values.
    img_w : int
        Image width in pixels.
    img_h : int
        Image height in pixels.

    Returns
    -------
    tuple
        (x1, y1, x2, y2) pixel coordinates.
    """
    _, x_c, y_c, w, h = box

    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)

    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def resize_with_padding(img, target_size=64):
    """
    Resize an image to target_size x target_size using padding
    to preserve aspect ratio.

    Parameters
    ----------
    img : np.ndarray
        Input cropped image.
    target_size : int
        Desired output dimension.

    Returns
    -------
    np.ndarray
        Padded and resized RGB image of shape (target_size, target_size, 3).
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded

# an optimized version of process_dataset_batches
def batch_generator(image_paths, labels_dir, batch_size, target_size):
    """Generate batches of data on the fly."""
    num_samples = len(image_paths)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_paths = [image_paths[i] for i in batch_indices]
        X_batch, Y_batch = process_dataset_batches(
            batch_paths, labels_dir, 
            starting_image_path_index=0,  # since we're passing specific paths
            target_size=target_size,
            batch_size=len(batch_paths)
        )
        X_batch = scale_data(X_batch, method='minmax')
        yield X_batch, Y_batch


def process_dataset_batches(image_paths, labels_dir, starting_image_path_index=0, target_size=64, batch_size = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a single batch of images and their corresponding bounding boxes.

    Parameters:
    ----------
        image_paths: list(str)
            List of image paths.
        labels_dir: str
            Directory containing YOLO label files.
        starting_image_path_index: int
            Index of the first image path to process.
        target_size: int
            Output size for each cropped object.
        batch_size: int
            Number of images to process at once.
      
    Returns:
    -------
        tuple
            (X, Y) where:            
            - X: Array of shape (target_size*target_size*3, m) where m is the number of cropped objects
            - Y: Array of shape (m, num_classes) containing multi-hot encoded labels
    """

    # validation of inputs
    if starting_image_path_index < 0:
        raise ValueError("starting_image_path_index must be non-negative")

    if target_size <= 0:
        raise ValueError("target_size must be positive")

    if not image_paths:
        raise ValueError("image_paths must not be empty")

    if batch_size is None:
        batch_size = len(image_paths)   
    
    all_crops = []
    all_labels = []
    end_index = min(starting_image_path_index + batch_size, len(image_paths))
    batch_paths = image_paths[starting_image_path_index:end_index]

    print(f"\nProcessing batch of {len(batch_paths)} images...", end='', flush=True)

    for i, img_path in enumerate(batch_paths):        
        if i % 100 == 0:
            print(f"\nProcessed {i} images...", end='', flush=True)
        else:
            print('.', end='', flush=True)

        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
        except Exception as e:
            print(f"\nError loading image {img_path}: {e}")
            continue

        label_path = os.path.join(
            labels_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        if not os.path.exists(label_path):
            print(f"[WARN] Skipping {img_path} - no label file")
            continue

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[ERROR] Failed to read labels for {img_path}: {e}")
            continue
              
        
        for line in lines:
            img_label = np.zeros(63, dtype=np.float32) 

            parts =  line.strip().split()
            if len(parts) != 5:
                print(f"[WARN] Invalid label format in {label_path}: {line.strip()}")
                continue

            class_id = int(parts[0])
            if class_id >= 63 or class_id < 0:
                print(f"[WARN] Invalid class ID {class_id} in {label_path}")
                continue
            

            box = list(map(float, parts))
            if len(box) != 5:
                print(f"[WARN] Invalid label format in {label_path}: {line.strip()}")
                continue    
            x1, y1, x2, y2 = yolo_to_pixel_coords(box, w, h)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue
            
            img_label[class_id] = 1.0
            crop_resized = resize_with_padding(crop, target_size)
            crop_flat = crop_resized.flatten()

            all_crops.append(crop_flat)
            all_labels.append(img_label.copy())

    data_matrix = np.array(all_crops).T  # shape: (12288, m)
    return data_matrix, np.array(all_labels)
    
    
# ============================================================
# Main Dataset Processing.
# NOTE: this function processes all images at once, which is memory intensive.
# ============================================================

def process_dataset(images_dir, labels_dir, target_size=64, max_images=-1):
    """
    Process all images and bounding boxes into a flattened dataset.

    Parameters
    ----------
    images_dir : str
        Directory containing training images.
    labels_dir : str
        Directory containing YOLO label files.
    target_size : int
        Output size for each cropped object.

    Returns
    -------
    tuple
        (X, Y) where:
        - X: Array of shape (target_size*target_size*3, m) where m is the number of cropped objects
        - Y: Array of shape (m, num_classes) containing multi-hot encoded labels
    """        
    print(f"Processing dataset from images_dir: {images_dir}, labels_dir: {labels_dir}")
    image_paths = sorted(glob(os.path.join(images_dir, "*.jpg")) +
                         glob(os.path.join(images_dir, "*.png")))
    
    if max_images > 0 and max_images < len(image_paths):
        image_paths = image_paths[:max_images]
        print(f"Limiting to loading {max_images} images")
                    
    print(f"Found {len(image_paths)} images in {images_dir}")
    print(f"Labels directory: {labels_dir}")

    print(f"Processing all images....")

    batch_size = 1000    
    X = []
    Y = []
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1} "
                 f"(images {batch_start}-{batch_end-1})")
        X_batch, Y_batch = process_dataset_batches(image_paths, labels_dir, starting_image_path_index = batch_start, target_size=target_size, batch_size = batch_size)

        if X_batch is None or Y_batch is None:
            continue
        X.append(X_batch)
        Y.append(Y_batch)
    
    if len(X) == 0 or len(Y) == 0:
        print("Warning: No valid data found in the dataset")
        return np.array([]).reshape((target_size*target_size*3, 0)), np.array([]).reshape((0, 63))  
    
    return np.concatenate(X, axis=1), np.concatenate(Y, axis=0)


def scale_data(data_matrix, method='minmax'):
    """
    Scale the data matrix using specified method.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Array of shape (features, samples)
    method : str
        Scaling method: 'minmax' or 'standard'
        
    Returns
    -------
    np.ndarray
        Scaled data matrix
    """
    if method == 'minmax':
        data_min = np.min(data_matrix, axis=1, keepdims=True)
        data_max = np.max(data_matrix, axis=1, keepdims=True)
        return (data_matrix - data_min) / (data_max - data_min + 1e-8)
    elif method == 'standard':
        data_mean = np.mean(data_matrix, axis=1, keepdims=True)
        data_std = np.std(data_matrix, axis=1, keepdims=True)
        return (data_matrix - data_mean) / (data_std + 1e-8)
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")

# ============================================================
# Test Code
# ============================================================
"""
if __name__ == "__main__":
    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process dataset with configurable number of images')
    parser.add_argument('--num-images', type=int, default=500,
                      help='Number of images to process (use -1 for all)')
    args = parser.parse_args()
    # Example small test using a few images
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images_root = os.path.join(project_root, "assets/produce_dataset/LVIS_Fruits_And_Vegetables")
    yaml_path = os.path.join(images_root, "data.yaml")
    
    # Load class information
    data = load_yaml_classes(yaml_path)
    num_classes = len(data["names"])
    train_images = os.path.join(images_root, data["train"])
    train_labels = train_images.replace("images", "labels")
    print(f"Loading {'all' if args.num_images == -1 else args.num_images} images...")
    X, Y = process_dataset(train_images, train_labels, target_size=64, max_images=args.num_images)
    print("\nDataset shapes:")
    print(f"Images (flattened): {X.shape}")  # (12288, m)
    print(f"Labels (one-hot): {Y.shape}")     # (m, 63)
    # Print first few labels
    print("\nFirst 5 image labels:")
    for i in range(min(5, Y.shape[0])):
        print(f"Image {i+1}: {Y[i].astype(int)}  (sum: {int(Y[i].sum())} classes)")
    # Show first crop to verify visually
    first_crop = X[:, 0].reshape(64, 64, 3)
    print("\nFirst crop min/max:", first_crop.min(), first_crop.max())
    
    # Test scaling
    X_scaled = scale_data(X, method='minmax')
    print("Scaled first crop min/max:", X_scaled[:, 0].min(), X_scaled[:, 0].max())
"""