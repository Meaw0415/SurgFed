import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Directory to save visualizations
save_dir = '/mnt/iMVR/zhengf/Medical-SAM2/vis_results'
os.makedirs(save_dir, exist_ok=True)

# Task, dataset, and client information
TASKS = [
    'seg',
    'seg',
    'seg',
    'dep',
    'dep',
]
DATASETS = [
    'Endovis2017',
    'Endovis2018',
    'AutoLaparo',
    'StereoMIS',
    'EndoNerf',
]
WEIGHTS_CLIENTS = [
    1,
    1,
    8,
    15,
    170,
]

def load_state_dicts(weights_paths):
    """
    Load the state dicts for each model from the given weights paths.
    
    :param weights_paths: List of file paths for the model weights.
    :return: A list of state dicts.
    """
    state_dicts = []
    for path in weights_paths:
        state_dict = torch.load(path)

        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove the 'module.' prefix if it exists
            if k.startswith('module.'):
                new_key = k[len('module.'):]  # Remove 'module.' from the key
            else:
                new_key = k
            new_state_dict[new_key] = v
        state_dicts.append(new_state_dict['model'])
    return state_dicts

def extract_common_layer_parameters(state_dicts, layer_types):
    """
    Extract common parameters from specified layers across all models.
    
    :param state_dicts: List of state dicts from each model.
    :param layer_types: List of layer names to extract.
    :return: A list of common parameter vectors for each model.
    """
    # Step 1: Collect parameter names for each model
    param_names_per_model = []
    for state_dict in state_dicts:
        # print(state_dict)
        param_names = set()
        for name, param in state_dict.items():
            if any(layer_type in name for layer_type in layer_types):
                param_names.add(name)
        param_names_per_model.append(param_names)

    # Step 2: Find common parameter names across all models
    common_param_names = set.intersection(*param_names_per_model)
    print(f'Common parameter names: {common_param_names}')
    # Step 3: Extract the common parameters and ensure consistency in vector size
    param_vectors = []
    for state_dict in state_dicts:
        params = []
        for name, param in state_dict.items():
            if name in common_param_names:  # Only extract common parameters
                params.append(param.flatten().float().cpu().numpy())  # Flatten parameters
        param_vectors.append(np.concatenate(params))  # Concatenate parameters to a single vector
    
    return param_vectors

def tsne_visualization(param_vectors, tasks, datasets, clients):
    """
    Perform t-SNE on the extracted parameters and visualize the result.
    
    :param param_vectors: List of parameter vectors for each model.
    :param tasks: List of task labels for each model (e.g., 'seg', 'dep').
    :param datasets: List of dataset names for each model.
    :param clients: List of client weights for each model.
    """
    # Convert list of param_vectors to numpy array
    param_vectors = np.array(param_vectors)
    
    # Perform t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42, perplexity=4)
    param_vectors_tsne = tsne.fit_transform(param_vectors)
    
    # Visualize the t-SNE results
    plt.figure(figsize=(8, 6))
    colors = ['red' if task == 'seg' else 'blue' for task in tasks]
    plt.scatter(param_vectors_tsne[:, 0], param_vectors_tsne[:, 1], c=colors)

    # Annotate with dataset names and client weights
    for i, (x, y) in enumerate(param_vectors_tsne):
        plt.text(x, y, f'{datasets[i]} ({clients[i]})', fontsize=8)
    
    plt.title('t-SNE Visualization of SAM and MEM Layer Parameters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(['Segmentation', 'Depth'])
    
    # Save the t-SNE plot
    tsne_save_path = os.path.join(save_dir, 'tsne_visualization.png')
    plt.savefig(tsne_save_path)
    plt.close()
    
    print(f't-SNE visualization saved to {tsne_save_path}')

def compute_similarity_matrix(param_vectors, datasets):
    """
    Compute the similarity (distance) matrix between models' parameters and visualize it with dataset labels.
    
    :param param_vectors: List of parameter vectors for each model.
    :param datasets: List of dataset names for each model.
    :return: A distance matrix.
    """
    # Convert list of param_vectors to numpy array
    param_vectors = np.array(param_vectors)
    
    # Compute the pairwise Euclidean distances between parameter vectors
    distance_matrix = squareform(pdist(param_vectors, metric='euclidean'))
    
    # Visualize the similarity matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Model Parameter Distance Matrix (SAM and MEM Layers)')
    
    # Set dataset names as x and y labels
    plt.xticks(ticks=np.arange(len(datasets)), labels=datasets, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(datasets)), labels=datasets)
    
    # Annotate each cell in the matrix with the similarity value
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            plt.text(j, i, f'{distance_matrix[i, j]:.2f}', ha='center', va='center', color='white')

    # Save the similarity matrix plot
    matrix_save_path = os.path.join(save_dir, 'similarity_matrix.png')
    plt.savefig(matrix_save_path)
    plt.close()
    
    print(f'Similarity matrix saved to {matrix_save_path}')
    
    return distance_matrix

def plot_parameter_difference(param_vectors, datasets):
    """
    Calculate and visualize the parameter differences between models in a 2D heatmap.
    
    :param param_vectors: List of parameter vectors for each model.
    :param datasets: List of dataset names for each model.
    """
    # Convert list of param_vectors to numpy array
    param_vectors = np.array(param_vectors)
    
    # Compute the pairwise Euclidean distances between parameter vectors
    difference_matrix = squareform(pdist(param_vectors, metric='euclidean'))
    
    # Visualize the difference matrix as a heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(difference_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Model Parameter Difference Heatmap (SAM and MEM Layers)')
    
    # Set dataset names as x and y labels
    plt.xticks(ticks=np.arange(len(datasets)), labels=datasets, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(datasets)), labels=datasets)
    
    # Annotate each cell in the heatmap with the difference value
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            plt.text(j, i, f'{difference_matrix[i, j]:.2f}', ha='center', va='center', color='white')

    # Save the heatmap plot
    heatmap_save_path = os.path.join(save_dir, 'parameter_difference_heatmap.png')
    plt.savefig(heatmap_save_path)
    plt.close()
    
    print(f'Parameter difference heatmap saved to {heatmap_save_path}')
    
    return difference_matrix

from sklearn.decomposition import PCA

def pca_visualization(param_vectors, tasks, datasets):
    """
    Perform PCA on the extracted parameters and visualize the result in a 2D scatter plot.
    
    :param param_vectors: List of parameter vectors for each model.
    :param tasks: List of task labels for each model (e.g., 'seg', 'dep').
    :param datasets: List of dataset names for each model.
    """
    # Convert list of param_vectors to numpy array
    param_vectors = np.array(param_vectors)
    
    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    param_vectors_pca = pca.fit_transform(param_vectors)
    
    # Visualize the PCA result as a 2D scatter plot
    plt.figure(figsize=(8, 6))
    colors = ['red' if task == 'seg' else 'blue' for task in tasks]
    plt.scatter(param_vectors_pca[:, 0], param_vectors_pca[:, 1], c=colors)
    
    # Annotate each point with the dataset name
    for i, (x, y) in enumerate(param_vectors_pca):
        plt.text(x, y, f'{datasets[i]}', fontsize=8)
    
    plt.title('PCA Visualization of SAM and MEM Layer Parameters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(['Segmentation', 'Depth'])
    
    # Save the PCA plot
    pca_save_path = os.path.join(save_dir, 'pca_visualization.png')
    plt.savefig(pca_save_path)
    plt.close()
    
    print(f'PCA visualization saved to {pca_save_path}')

from sklearn.manifold import MDS

def mds_visualization(similarity_matrix, tasks, datasets):
    """
    Perform MDS based on the similarity matrix and visualize the result in a 2D scatter plot.
    
    :param similarity_matrix: A distance matrix (similarity matrix) between models.
    :param tasks: List of task labels for each model (e.g., 'seg', 'dep').
    :param datasets: List of dataset names for each model.
    """
    # Perform MDS to reduce the similarity matrix to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_result = mds.fit_transform(similarity_matrix)
    
    # Visualize the MDS result as a 2D scatter plot
    plt.figure(figsize=(8, 6))
    colors = ['red' if task == 'seg' else 'blue' for task in tasks]
    plt.scatter(mds_result[:, 0], mds_result[:, 1], c=colors)
    
    # Annotate each point with the dataset name
    for i, (x, y) in enumerate(mds_result):
        plt.text(x, y, f'{datasets[i]}', fontsize=8)
    
    plt.title('MDS Visualization Based on Model Similarity')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.legend(['Segmentation', 'Depth'])
    
    # Save the MDS plot
    mds_save_path = os.path.join(save_dir, 'mds_visualization.png')
    plt.savefig(mds_save_path)
    plt.close()
    
    print(f'MDS visualization saved to {mds_save_path}')


def main():
    # Model paths
    models_paths = [
        '/mnt/iMVR/zhengf/Medical-SAM2/logs/Endovis2017_2024_10_12_22_43_25/Model/latest_epoch.pth',
        '/mnt/iMVR/zhengf/Medical-SAM2/logs/Endovis2018_2024_10_13_00_18_31/Model/latest_epoch.pth',
        '/mnt/iMVR/zhengf/Medical-SAM2/logs/AutoLaparo_new_2024_10_13_02_16_22/Model/latest_epoch.pth',
        '/mnt/iMVR/zhengf/Medical-SAM2/logs/StereoMIS_2024_10_13_05_31_14/Model/latest_epoch.pth',
        '/mnt/iMVR/zhengf/Medical-SAM2/logs/EndoNerf_2024_10_13_05_26_11/Model/latest_epoch.pth',
    ]
    
    # Load state dicts
    state_dicts = load_state_dicts(models_paths)
    
    # Extract SAM and MEM layer parameters, only common parameters across models
    param_vectors = extract_common_layer_parameters(state_dicts, layer_types=['sam_mask_decoder', 'obj_ptr_proj', 'memory_encoder', 'memory_attention', 'mask_downsample'])

    # Perform t-SNE visualization and save the result
    # tsne_visualization(param_vectors, TASKS, DATASETS, WEIGHTS_CLIENTS)
    pca_visualization(param_vectors, TASKS, DATASETS)
    mds_visualization(param_vectors, TASKS, DATASETS)
    # Compute and visualize the similarity matrix between the models, and save the result
    similarity_matrix = compute_similarity_matrix(param_vectors, DATASETS)

if __name__ == '__main__':
    main()
