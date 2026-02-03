import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

from .ResNet50 import ResNet50
from ..mnist_signs_loader import prepare_sign_mnist_data
from .ResNet_utils import convert_to_one_hot
from src.configs import EnvironmentConfig
from .predict import predict_with_image


def train_model(force_train: bool = False):
    """
    Train ResNet50 model on SignMNIST dataset
    Args:
     force_train: bool, whether to force training even if model exists
    Returns:
     model: ResNet50 model
    """
    env_config = EnvironmentConfig()

    # load model if exists
    if force_train or not Path(env_config.get_saved_model_dir() / 'resnet50.keras').exists():
        sign_path = env_config.get_sign_path()
        (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig), classes = prepare_sign_mnist_data(train_path=f'{sign_path}/sign_mnist_train.csv', test_path=f'{sign_path}/sign_mnist_test.csv', dataset_dir=sign_path)    
        
        num_of_classes = len(np.unique(Y_train_orig)) + 1
        model = ResNet50(input_shape = (28 , 28, 1), classes = num_of_classes, training=True)
        opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        print ("number of training examples = " + str(X_train_orig.shape[0]))
        print ("number of test examples = " + str(X_test_orig.shape[0]))
        print ("X_train original shape: " + str(X_train_orig.shape))
        print ("Y_train original shape: " + str(Y_train_orig.shape))
        print ("X_test original shape: " + str(X_test_orig.shape))
        print ("Y_test original shape: " + str(Y_test_orig.shape))
        
        print(f"First 5 values of Y_train_orig: {Y_train_orig[:5]}")
        print(f"Y_train_orig.max(): {Y_train_orig.max()}")
        

        print ("num of classes: " + str(num_of_classes))
        
        # Convert training and test labels to one hot matrices
        Y_train = convert_to_one_hot(Y_train_orig, num_of_classes).T
        Y_test = convert_to_one_hot(Y_test_orig, num_of_classes).T

        print ("number of training examples = " + str(X_train_orig.shape[0]))
        print ("number of test examples = " + str(X_test_orig.shape[0]))
        print ("X_train shape: " + str(X_train_orig.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_test shape: " + str(X_test_orig.shape))
        print ("Y_test shape: " + str(Y_test.shape))
        print(f"Training model...")

        # add callbacks with checkpoints and early stopping
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), # stop training when validation loss stops improving
            tf.keras.callbacks.ModelCheckpoint(env_config.get_saved_model_dir() / 'resnet50_checkpoints.keras', monitor='val_loss', save_best_only=True) # save best model
        ]
        model.fit(X_train_orig, Y_train, validation_data=(X_test_orig, Y_test), epochs=10, batch_size=32, callbacks=callbacks)
        model.save(env_config.get_saved_model_dir() / 'resnet50.keras')

        # validate model on test set
        model.evaluate(X_test_orig, Y_test)
        model_history = model.history
        plt.plot(model_history.history['accuracy'], label='accuracy')
        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    else:
        model = load_model(env_config.get_saved_model_dir() / 'resnet50.keras')
        print(f"Model loaded from {env_config.get_saved_model_dir() / 'resnet50.keras'}")
    
    # predict with image
    image_path = env_config.get_sign_path() / 'my_image.jpg'
    if image_path.exists():
        predict_with_image(image_path, model, input_shape=(28, 28))
    else:
        print(f"Image {image_path} does not exist")
    
    return model
  