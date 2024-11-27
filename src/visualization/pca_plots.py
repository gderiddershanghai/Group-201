from sklearn.decomposition import PCA
import pandas as pd
from features.bert_embeddings import BertFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_top_2_pca_embeddings(df, embedding_methods=['mean', '[CLS]', 'pooler_output']):
    """
    Computes the top 2 PCA embeddings for a DataFrame using specified embedding methods.
    
    Parameters:
    - df: DataFrame containing the data.
    - embedding_methods: List of embedding types to compute PCA for (e.g., 'mean', '[CLS]', 'pooler_output').
    
    Returns:
    - Dictionary where keys are embedding method names and values are DataFrames of PCA components.
    """
    extractor = BertFeatureExtractor()
    pca_results = {}

    for method in embedding_methods:
        embeddings = extractor.transform(df, embedding_type=method)
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(embeddings)
        pca_results[method] = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    
    return pca_results

def plot_pca_embeddings(
    pca_results, df, domain_score_column, essay_set, chunk_flag, save_flag=False, binary=False
):
    """
    Plots scatter plots for the top 2 PCA components for multiple embedding methods in a single row.
    
    Parameters:
    - pca_results: Dictionary of PCA results from get_top_2_pca_embeddings.
    - df: Original DataFrame containing the domain scores.
    - domain_score_column: Column name in df with the target scores.
    - essay_set: The essay set identifier (e.g., 'Essay Set 1').
    - chunk_flag: Boolean indicating whether the data represents chunks or full essays.
    - binary: Boolean indicating whether to convert scores to 'low' and 'high' by removing the middle 25%.
    - save_flag: Boolean indicating whether to save the entire plot as an image.
    """
    sns.set_style("whitegrid")  # Set Seaborn white grid style
    df = df.reset_index(drop=True)
    if binary:
        # Calculate thresholds for the middle 25% range
        lower_bound = df[domain_score_column].quantile(0.4)
        upper_bound = df[domain_score_column].quantile(0.7)
        
        # Assign labels: 'low' for below lower bound, 'high' for above upper bound, and drop middle
        df['binary_label'] = df[domain_score_column].apply(
            lambda x: 'low' if x < lower_bound else 'high' if x > upper_bound else None
        )
        # Remove rows with None (middle scores)
        binary_df = df.dropna(subset=['binary_label']).copy()
        binary_df['binary_label'] = binary_df['binary_label'].map({'low': 0, 'high': 1})
        
        # Filter PCA results to match the updated DataFrame
        pca_results = {method: pca.loc[binary_df.index] for method, pca in pca_results.items()}

        binary_df.reset_index(drop=True, inplace=True)

    else:
        binary_df = df.copy()

    chunk_status = "Chunks" if chunk_flag else "Full Essays"
    plot_title = f"PCA Scatter Plots for {essay_set} ({chunk_status})"
    
    # Create a single row of three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(plot_title, fontsize=16)
    
    for ax, (method, pca_df) in zip(axes, pca_results.items()):
        scatter = ax.scatter(
            x=pca_df['PCA1'],
            y=pca_df['PCA2'],
            c=binary_df['binary_label'] if binary else binary_df[domain_score_column],
            cmap='viridis',
            alpha=0.7
        )
        ax.set_title(f"{method} Embeddings")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2" if ax == axes[0] else "")
        ax.grid(True)
    
    # Add a single colorbar for all plots
    cbar_label = 'Low (≤40th percentile) and High (≥70th percentile)' if binary else 'Essay Score'
    fig.colorbar(scatter, ax=axes, location='right', shrink=0.7, label=cbar_label)

    if save_flag:
        save_dir = "/home/ginger/code/gderiddershanghai/Group-201/plots"
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        file_name = f"{essay_set.replace(' ', '_').lower()}_{chunk_status.lower()}_binary_{binary}_pca_scatter.png"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {file_path}")
    plt.show()
