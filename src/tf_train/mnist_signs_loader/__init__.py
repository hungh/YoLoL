"""
MNIST Sign Language Dataset Loader
Download the dataset from kaggle and load it into a numpy array

Usage:
Make sure you have kaggle installed and logged (https://www.kaggle.com/docs/api)
```
kaggle datasets download -d datamunge/sign-language-mnist
```

"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os

assets = os.path.join(os.path.dirname(__file__), 'assets')

def prepare_sign_mnist_data(train_path, test_path, dataset_dir=os.path.join(os.path.dirname(__file__), 'assets')):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # check if the files already exist
    if os.path.exists(os.path.join(dataset_dir, 'x_train.npy')):
        print("Files already exist, loading from .npy files.")
        X_train = np.load(os.path.join(dataset_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(dataset_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))
        classes = np.unique(y_train)
        return (X_train, y_train), (X_test, y_test), classes    

    # 1. Load CSVs
    print("Loading CSVs...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Extract and Preprocess
    # Labels are first column, pixels are the rest
    y_train = train_df.iloc[:, 0].values
    y_test = test_df.iloc[:, 0].values
    
    # Reshape to (Samples, 28, 28, 1) and normalize to [0, 1]
    X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    classes = np.unique(y_train)

    np.save(os.path.join(dataset_dir, 'x_train.npy'), X_train)
    np.save(os.path.join(dataset_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(dataset_dir, 'x_test.npy'), X_test)
    np.save(os.path.join(dataset_dir, 'y_test.npy'), y_test)
    print("Saved binary .npy files.")

    return (X_train, y_train), (X_test, y_test), classes