import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from .ResNet_utils import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform

from .blocks_definition import identity_block, convolutional_block

"""
The implementation of ResNet50 is based on the original paper: https://arxiv.org/abs/1512.03385
NOTE: The input shape is (64, 64, 3) for the original paper, but we use (28, 28, 1) for SignMNIST dataset (MNIST dataset for sign language)
We are using GlobalAveragePooling2D instead of AveragePooling2D as the input shape is smaller than 64x64 to avoid the loss of information (negative impact on performance).

Also ResNet50 is also implemented in Keras, but we use our own implementation to have more control over the training process. Here is an example of ResNet50 using cifar dataset: https://keras.io/api/applications/resnet50/
```
import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)
```
TO allow training on Apple Silicon:
```
tf.config.set_device_policy(tf.DevicePolicy.WARNING)
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
try:
    # Try to enable MPS GPU acceleration
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[0], True
    )
    print("MPS GPU enabled")
except:
    print("MPS GPU not available, using CPU instead")
```

"""
def ResNet50(input_shape = (64, 64, 3), classes = 6, training=False):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1, training=training)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], s = 2, training=training)
    
    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2, training=training)
    
    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2, training=training)
    
    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [512, 512, 2048], training=training)
    X = identity_block(X, 3, [512, 512, 2048], training=training)

    # AVGPOOL
    if input_shape[0] < 64:
        X = GlobalAveragePooling2D()(X)
    else:
        X = AveragePooling2D(pool_size=(2,2))(X)
        # output layer
        X = Flatten()(X)

    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model