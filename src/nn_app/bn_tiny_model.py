"""
Batch Normalization Tiny Model
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..classic_nn.batch_norm import forward_and_backward_propagation, initialize_parameters_deep, custom_model_forward
# from ..classic_nn import forward_and_backward_propagation, initialize_parameters_deep

# using skit-learn to generate a binary classification data set
# using skit-learn to train a binary classification model with batch normalization
# using skit-learn to evaluate the model
# using skit-learn to plot the results

def generate_binary_classification_data():
    X, Y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    return X, Y

def train_model():
    X, Y = generate_binary_classification_data()
    # split the dataset into train and dev
    from sklearn.model_selection import train_test_split
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    # normalize the data
    from sklearn.preprocessing import StandardScaler
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
    layers_dims = [2, 4, 1]
    activations = ["relu", "relu", "linear"]
    lambda_reg = 0.01
    number_of_classes = 1
    # initialize parameters
    parameters = initialize_parameters_deep(layers_dims)
    
    # train the model
    costs = []
    for i in range(num_iterations):
        # forward and backward pass        
        cost, _, parameters = forward_and_backward_propagation(X_train, Y_train, parameters, activations, learning_rate=learning_rate,num_classes=number_of_classes, last_activation="sigmoid", lambda_reg=lambda_reg)
        costs.append(cost)      

        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    

    evaluate_model(X_dev, Y_dev, parameters, activations, num_classes=number_of_classes, last_activation="sigmoid", lambda_reg=lambda_reg)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # plot the cost
    ax1.plot(costs)
    ax1.set_title("Cost vs Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")
    
    # plot decision boundary
    plot_decision_boundary(X_dev, Y_dev, parameters, activations, number_of_classes, ax2)


    plt.show()


def predict(X, parameters, activations, num_classes):
    A, _ = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(num_classes == 1))
    return A


def plot_decision_boundary(X, Y, parameters, activations, num_classes, ax):
    # Transpose X back to (n_samples, n_features) for plotting
    X_plot = X.T  # From (n_features, n_samples) to (n_samples, n_features)
    
    # Reshape Y to 1D if needed
    if Y.ndim > 1:
        Y_plot = Y.reshape(-1)  # From (n_classes, n_samples) to (n_samples,)
    else:
        Y_plot = Y
    
    # Create a mesh grid
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict on mesh grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parameters, activations, num_classes)
    Z = Z.reshape(xx.shape)
    
    # Plot contour
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=Y_plot, edgecolors='k')    


def evaluate_model(X_dev, Y_dev, parameters, activations, num_classes=1, last_activation="sigmoid", lambda_reg=0.01):
    A, _ = custom_model_forward(X_dev, parameters, activations, num_classes, apply_sigmoid=(last_activation == "sigmoid"))

    # Reshape Y_dev from (1, n_samples) to (n_samples,)
    Y_dev_flat = Y_dev.reshape(-1)
    A_flat = A.reshape(-1)
    
    # calculate accuracy
    accuracy = np.mean((A_flat > 0.5) == Y_dev_flat)
    print(f"Accuracy on dev set: {accuracy}")

    # calculate f1 score    
    accuracy = accuracy_score(Y_dev_flat, A_flat > 0.5)
    precision = precision_score(Y_dev_flat, A_flat > 0.5)
    recall = recall_score(Y_dev_flat, A_flat > 0.5)
    f1 = f1_score(Y_dev_flat, A_flat > 0.5)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    train_model()


    
    