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


def multi_label_metrics(Y_pred, Y_true):
    from sklearn.metrics import hamming_loss, f1_score
    
    # Convert to binary predictions
    Y_pred_binary = (Y_pred > 0.5).astype(int)
    
    # Exact match ratio (strict accuracy)
    exact_match = np.mean(np.all(Y_pred_binary == Y_true, axis=0))
    
    # Hamming loss (fraction of wrong labels)
    hamming = hamming_loss(Y_true, Y_pred_binary)
    
    # F1 score (micro-averaged)
    f1 = f1_score(Y_true, Y_pred_binary, average='micro')
    
    return {
        'exact_match': exact_match,
        'hamming_loss': hamming,
        'f1_score': f1,
        'avg_predicted': np.mean(Y_pred_binary),
        'avg_actual': np.mean(Y_true)
    }

def plot_dataset(X, Y, max_samples=1000):
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
    Y = Y.T.argmax(axis=1)  # Convert one-hot to class indices
    
    # Limit number of samples for better visualization
    if X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
    
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


def plot_costs(costs, title="Training Costs"):
    """
    Plot the cost function over epochs.
    
    Parameters:
    -----------
    costs : list
        List of cost values over epochs
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def mini_batch_generator(X_train, Y_train, batch_size):
    """
    Generate batches of data for training.
    """
    for i in range(0, X_train.shape[1], batch_size):
        yield X_train[:, i:i+batch_size], Y_train[:, i:i+batch_size]

        