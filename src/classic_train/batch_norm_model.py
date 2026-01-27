"""
Batch Normalization Model
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ..classic_nn.batch_norm import gradient_descent, initialize_parameters_deep
from ..classic_nn.optimizers import OptimizerFactory
from ..classic_nn.data_utils import generate_binary_classification_data
from ..classic_nn.utils import plot_decision_boundary
from ..classic_nn.utils.eval import evaluate_model, predict


def train_model():
    X, Y = generate_binary_classification_data()
    # split the dataset into train and dev
    from sklearn.model_selection import train_test_split
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    # normalize the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_dev = sc.transform(X_dev)

    # reshape X
    X_train = X_train.T
    X_dev = X_dev.T
 
    # reshape Y
    Y_train = Y_train.reshape(1, -1)
    Y_dev = Y_dev.reshape(1, -1)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_dev shape: {Y_dev.shape}")

    # hyperparameters
    learning_rate = 0.01
    num_iterations = 1000
    layers_dims = [X_train.shape[0], 8, 4, 1]
    activations = ["relu", "relu", "sigmoid"]
    lambda_reg = 0.01
    number_of_classes = 1
    # momentum
    beta = 0.9
    optimizer = OptimizerFactory("momentum").get_optimizer(beta=beta)
    # initialize parameters
    parameters = initialize_parameters_deep(layers_dims, optimizer)
    
    # train the model
    costs = []
    for i in range(num_iterations):
        # forward and backward pass        
        cost, _, parameters = gradient_descent(X_train, Y_train, parameters, activations, learning_rate=learning_rate,num_classes=number_of_classes, last_activation="sigmoid", lambda_reg=lambda_reg)
        costs.append(cost)      

        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    print(f"Training metrics:")
    evaluate_model(X_train, Y_train, parameters, activations, num_classes=number_of_classes, last_activation="sigmoid")

    print(f"\n\nDev metrics:")
    evaluate_model(X_dev, Y_dev, parameters, activations, num_classes=number_of_classes, last_activation="sigmoid")
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # plot the cost
    ax1.plot(costs)
    ax1.set_title("Cost vs Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")
    
    # plot decision boundary
    plot_decision_boundary(X_dev, Y_dev, predict, parameters, activations, number_of_classes, ax2)
    plt.show()


    
    