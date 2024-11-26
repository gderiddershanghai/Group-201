from sklearn.decomposition import PCA
import numpy as np

def run_pca_on_train_and_transform(train_embeddings, val_embeddings, variance_threshold=0.90):
    """
    Performs PCA on the training embeddings and transforms both the training and validation embeddings,
    retaining the top components that explain the specified variance threshold.

    Parameters:
    train_embeddings (numpy.ndarray): Training embeddings (2D array of shape (n_train_samples, n_features)).
    val_embeddings (numpy.ndarray): Validation embeddings (2D array of shape (n_val_samples, n_features)).
    variance_threshold (float): The cumulative variance threshold to retain components (default is 0.90).

    Returns:
    tuple: Transformed training and validation embeddings as numpy arrays and the fitted PCA object.
    """
    # Ensure inputs are numpy arrays
    train_embeddings = np.array(train_embeddings)
    val_embeddings = np.array(val_embeddings)
    
    # Initialize PCA
    pca = PCA()
    
    # Fit PCA on training embeddings
    pca.fit(train_embeddings)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components to retain to reach the variance threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Perform PCA with the selected number of components
    pca = PCA(n_components=n_components)
    train_transformed = pca.fit_transform(train_embeddings)
    val_transformed = pca.transform(val_embeddings)  # Transform validation embeddings
    
    return train_transformed, val_transformed, pca
