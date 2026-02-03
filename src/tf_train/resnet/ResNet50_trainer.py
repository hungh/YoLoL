
from ResNet50 import ResNet50
from mnist_signs_loader import prepare_sign_mnist_data
from RestNet_utils import convert_to_one_hot


def train_model():
    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig), classes = prepare_sign_mnist_data(train_path='datasets/sign_mnist_train.csv', test_path='datasets/sign_mnist_test.csv', dataset_dir='datasets')
    
    model = ResNet50()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, len(classes)).T
    Y_test = convert_to_one_hot(Y_test_orig, len(classes)).T

    model.fit(X_train_orig, Y_train, epochs=10, batch_size=32)

    model.save('saved_model/resnet50.h5')

    print ("number of training examples = " + str(X_train_orig.shape[0]))
    print ("number of test examples = " + str(X_test_orig.shape[0]))
    print ("X_train shape: " + str(X_train_orig.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test_orig.shape))
    print ("Y_test shape: " + str(Y_test.shape))