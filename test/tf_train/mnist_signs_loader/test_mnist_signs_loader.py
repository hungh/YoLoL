import os
import tempfile
import numpy as np
from src.tf_train.mnist_signs_loader import prepare_sign_mnist_data

def test_prepare_sign_mnist_data():
    train_path = "test/resources/assets/mnist/signs//sign_mnist_train.csv"
    test_path = "test/resources/assets/mnist/signs//sign_mnist_test.csv"
    dataset_dir = "test/resources/assets/mnist/signs"
    (X_train, y_train), (X_test, y_test), classes = prepare_sign_mnist_data(train_path=train_path, test_path=test_path, dataset_dir=dataset_dir)

    # Test data types
    assert X_train.dtype == 'float32'
    assert X_test.dtype == 'float32'
    assert y_train.dtype in ['int64', 'int32']
    assert y_test.dtype in ['int64', 'int32']
    
    # Test normalization (should be between 0 and 1)
    assert 0 <= X_train.min() <= X_train.max() <= 1.0
    assert 0 <= X_test.min() <= X_test.max() <= 1.0
    
    # Test image dimensions
    assert X_train.shape[1:] == (28, 28, 1)  # Height, Width, Channels
    assert X_test.shape[1:] == (28, 28, 1)


def test_prepare_sign_mnist_data_caching():
    """Test that caching mechanism works correctly"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = "test/resources/assets/mnist/signs//sign_mnist_train.csv"
        test_path = "test/resources/assets/mnist/signs//sign_mnist_test.csv"
        
        # First call - should process and save
        (X_train1, y_train1), (X_test1, y_test1), classes1 = prepare_sign_mnist_data(
            train_path=train_path, test_path=test_path, dataset_dir=temp_dir
        )
        
        # Check that .npy files were created
        assert os.path.exists(os.path.join(temp_dir, 'x_train.npy'))
        assert os.path.exists(os.path.join(temp_dir, 'y_train.npy'))
        assert os.path.exists(os.path.join(temp_dir, 'x_test.npy'))
        assert os.path.exists(os.path.join(temp_dir, 'y_test.npy'))
        
        # Second call - should load from cache
        (X_train2, y_train2), (X_test2, y_test2), classes2 = prepare_sign_mnist_data(
            train_path=train_path, test_path=test_path, dataset_dir=temp_dir
        )
        
        # Results should be identical
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_test1, y_test2)
        assert np.array_equal(classes1, classes2)


def test_prepare_sign_mnist_data_caching():
    """Test that caching mechanism works correctly"""

    
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = "test/resources/assets/mnist/signs//sign_mnist_train.csv"
        test_path = "test/resources/assets/mnist/signs//sign_mnist_test.csv"
        
        # First call - should process and save
        (X_train1, y_train1), (X_test1, y_test1), classes1 = prepare_sign_mnist_data(
            train_path=train_path, test_path=test_path, dataset_dir=temp_dir
        )
        
        # Check that .npy files were created
        assert os.path.exists(os.path.join(temp_dir, 'x_train.npy'))
        assert os.path.exists(os.path.join(temp_dir, 'y_train.npy'))
        assert os.path.exists(os.path.join(temp_dir, 'x_test.npy'))
        assert os.path.exists(os.path.join(temp_dir, 'y_test.npy'))
        
        # Second call - should load from cache
        (X_train2, y_train2), (X_test2, y_test2), classes2 = prepare_sign_mnist_data(
            train_path=train_path, test_path=test_path, dataset_dir=temp_dir
        )
        
        # Results should be identical
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_test1, y_test2)
        assert np.array_equal(classes1, classes2)