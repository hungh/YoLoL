import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.torch_train.dataset.ASLDataSet import ASLDataSet


def create_mock_csv(csv_path, num_samples=100, num_classes=24):
    """Create a mock ASL dataset CSV file for testing"""
    np.random.seed(42)  # For reproducible tests
    
    # Generate random labels (0-23 for ASL letters A-Y excluding J, Z)
    labels = np.random.randint(0, num_classes, num_samples)
    
    # Generate random pixel data (784 pixels for 28x28 images)
    pixels = np.random.randint(0, 256, (num_samples, 784))
    
    # Create DataFrame
    columns = ['label'] + [f'pixel_{i}' for i in range(784)]
    data = np.column_stack([labels, pixels])
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(csv_path, index=False, header=True)
    return csv_path


def test_asl_dataset_init():
    """Test ASLDataSet initialization"""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=50)
        
        dataset = ASLDataSet(csv_path)
        
        # Test basic properties
        assert len(dataset) == 50
        assert hasattr(dataset, 'data')
        assert hasattr(dataset, 'labels')
        assert hasattr(dataset, 'images')
        assert hasattr(dataset, 'transform')


def test_asl_dataset_data_types():
    """Test that data types are correct"""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=10)
        
        dataset = ASLDataSet(csv_path)
        
        # Test data types
        assert dataset.labels.dtype == np.int64
        assert dataset.images.dtype == np.float32
        assert dataset.images.shape == (10, 28, 28)  # (samples, height, width)


def test_asl_dataset_getitem_no_transform():
    """Test __getitem__ without transforms"""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=5)
        
        dataset = ASLDataSet(csv_path)
        
        # Test getting items
        img, label = dataset[0]
        
        # Test return types
        assert isinstance(img, Image.Image)
        assert isinstance(label, np.int64)
        assert img.mode == 'L'  # Grayscale
        assert img.size == (28, 28)
        assert 0 <= label < 24


def test_asl_dataset_getitem_with_transform():
    """Test __getitem__ with transforms"""
    
    
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=5)
        
        # Define simple transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        
        dataset = ASLDataSet(csv_path, transform=transform)
        
        # Test getting items
        img, label = dataset[0]
        
        # Test return types after transform
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, np.int64)
        assert img.shape == (3, 224, 224)  # (channels, height, width)
        assert img.dtype == torch.float32
        assert 0 <= img.min() <= img.max() <= 1.0  # Normalized


def test_asl_dataset_dataloader_compatibility():
    """Test that dataset works with DataLoader"""    
    
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=20)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        
        dataset = ASLDataSet(csv_path, transform=transform)
        
        # Test with DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test iteration
        for batch_imgs, batch_labels in dataloader:
            assert batch_imgs.shape[0] <= 4  # Batch size
            assert batch_imgs.shape[1:] == (3, 224, 224)  # Image shape
            assert batch_labels.shape[0] <= 4  # Batch size
            assert batch_imgs.dtype == torch.float32
            assert batch_labels.dtype == torch.int64
            break  # Just test first batch


def test_asl_dataset_label_range():
    """Test that labels are in expected range for ASL"""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=100, num_classes=24)
        
        dataset = ASLDataSet(csv_path)
        
        # Test label range (should be 0-23 for ASL A-Y excluding J, Z)
        unique_labels = np.unique(dataset.labels)
        assert unique_labels.min() >= 0
        assert unique_labels.max() <= 23
        assert len(unique_labels) <= 24


def test_asl_dataset_pixel_normalization():
    """Test pixel values are properly normalized"""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=10)
        
        dataset = ASLDataSet(csv_path)
        
        # Test raw pixel values (should be 0-255)
        assert dataset.images.min() >= 0.0
        assert dataset.images.max() <= 255.0
        
        # Test PIL conversion in __getitem__
        img, _ = dataset[0]
        img_array = np.array(img)
        assert img_array.min() >= 0
        assert img_array.max() <= 255


def test_asl_dataset_reproducibility():
    """Test that dataset returns consistent results"""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        create_mock_csv(csv_path, num_samples=10)
        
        dataset1 = ASLDataSet(csv_path)
        dataset2 = ASLDataSet(csv_path)
        
        # Test same index returns same data
        img1, label1 = dataset1[0]
        img2, label2 = dataset2[0]
        
        assert label1 == label2
        # PIL images should be identical
        assert np.array_equal(np.array(img1), np.array(img2))


def test_asl_dataset_edge_cases():
    """Test edge cases"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with single sample
        csv_path = os.path.join(temp_dir, "single_sample.csv")
        create_mock_csv(csv_path, num_samples=1)
        
        dataset = ASLDataSet(csv_path)
        assert len(dataset) == 1
        
        img, label = dataset[0]
        assert isinstance(img, Image.Image)
        assert isinstance(label, np.int64)


def test_asl_dataset_with_real_data():
    """Test with actual ASL dataset if available"""
    # This test will only run if the real dataset exists
    real_data_path = "assets/mnist/signs/sign_mnist_train.csv"
    
    if os.path.exists(real_data_path):
        dataset = ASLDataSet(real_data_path)
        
        # Test basic properties
        assert len(dataset) > 0
        assert dataset.images.shape[1:] == (28, 28)
        
        # Test getting a sample
        img, label = dataset[0]
        assert isinstance(img, Image.Image)
        assert isinstance(label, np.int64)
        assert img.mode == 'L'
        assert img.size == (28, 28)
    else:
        pytest.skip("Real ASL dataset not found")


if __name__ == "__main__":
    # Run tests manually
    test_asl_dataset_init()
    test_asl_dataset_data_types()
    test_asl_dataset_getitem_no_transform()
    test_asl_dataset_getitem_with_transform()
    test_asl_dataset_dataloader_compatibility()
    test_asl_dataset_label_range()
    test_asl_dataset_pixel_normalization()
    test_asl_dataset_reproducibility()
    test_asl_dataset_edge_cases()
    test_asl_dataset_with_real_data()
    print("All tests passed!")