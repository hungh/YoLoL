"""
Utility functions for neural networks
"""
import numpy as np
import matplotlib.pyplot as plt

def get_num_layers(parameters: dict) -> int:
    """
    Get the number of layers from parameters dictionary.
    
    Args:
        parameters: dictionary containing parameters
        
    Returns:
        Number of layers
    """
    if '_num_layers' in parameters:
        return parameters['_num_layers']

    layers = len([k for k in parameters.keys() if k.startswith('W')])
    parameters['_num_layers'] = layers
    return layers


def pca_plot_dataset(X, Y, max_samples=1000):
    """
    Plot a 2D projection of the dataset using PCA for visualization.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features of shape (n_features, n_samples)
    Y : numpy.ndarray
        Labels of shape (n_classes, n_samples)
    max_samples : int, optional
        Maximum number of samples to plot (for better performance)
    """
    from sklearn.decomposition import PCA
    
    # Transpose to (n_samples, n_features) for PCA
    X = X.T

    if Y.shape[0] == 1:
        Y = Y.T.flatten()
    else:
        Y = Y.T.argmax(axis=1)  # Convert one-hot to class indices
    
    # Limit number of samples for better visualization
    if X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
    
    # if X is 2D or less, don't reduce it
    if X.shape[1] <= 2:
        X_pca = X
    else:
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, 
                         cmap='viridis', alpha=0.6, 
                         edgecolors='w', s=40)
    plt.colorbar(scatter, label='Class')
    plt.title('2D PCA Projection of the Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_costs(costs, title="Training Costs", ax=None):
    """
    Plot the cost function over epochs.
    
    Parameters:
    -----------
    costs : list
        List of cost values over epochs
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.plot(costs)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_decision_boundary(X, Y, predict_func, parameters, activations, num_classes, ax):
    """
    Plot the decision boundary for the model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Training features of shape (n_features, n_samples)
    Y : numpy.ndarray
        Training labels of shape (n_classes, n_samples)
    predict_func : callable
        Function to predict labels
    parameters : dict
        Dictionary of model parameters
    activations : list
        List of activation functions for each layer
    num_classes : int
        Number of classes
    ax : matplotlib.axes.Axes
        Axes object to plot on
    """
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
    Z = predict_func(np.c_[xx.ravel(), yy.ravel()].T, parameters, activations, num_classes)
    Z = Z.reshape(xx.shape)
    
    # Plot contour
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=Y_plot, edgecolors='k')  

        