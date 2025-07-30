import copy
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def extract_common_parameters(model, common_layers, exclude_layers):
    """
    Extract common layer parameters from a model, excluding task-specific parameters.
    :param model: A model from which to extract parameters.
    :param common_layers: List of common layers (e.g., SAM and MEM layers).
    :param exclude_layers: List of task-specific layers to exclude.
    :return: Flattened parameter vector (numpy array) for the common layers.
    """
    params = []
    for name, param in model.state_dict().items():
        if any(layer_name in name for layer_name in common_layers) and not any(exclude_name in name for exclude_name in exclude_layers):
            params.append(param.flatten().cpu().numpy())  # Flatten the parameters and convert to numpy
    return np.concatenate(params)  # Concatenate all extracted parameters into a single vector


def tsne_visualization(models, tasks, common_layers, exclude_layers_per_task):
    """
    Perform t-SNE visualization on the extracted parameters from the models.
    :param models: List of models to visualize.
    :param tasks: List of corresponding tasks ('seg' or 'depth') for each model.
    :param common_layers: Layers that are shared across tasks.
    :param exclude_layers_per_task: A dictionary mapping tasks to task-specific layers to exclude.
    """
 
    param_vectors = []
    for i, model in enumerate(models):
        exclude_layers = exclude_layers_per_task[tasks[i]]
        param_vector = extract_common_parameters(model, common_layers, exclude_layers)
        param_vectors.append(param_vector)

    # 使用t-SNE将参数降维到2D空间
    tsne = TSNE(n_components=2, random_state=42)
    param_vectors_tsne = tsne.fit_transform(param_vectors)

    # 可视化t-SNE结果
    plt.figure(figsize=(8, 6))
    plt.scatter(param_vectors_tsne[:, 0], param_vectors_tsne[:, 1], c=['red' if task == 'seg' else 'blue' for task in tasks])
    plt.title('t-SNE Visualization of Model Parameters (Seg vs Depth)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(['Segmentation', 'Depth'])
    plt.show()


# 计算模型参数的相似度矩阵
def compute_similarity_matrix(param_vectors):
    """
    Compute the similarity (distance) matrix between models' parameters.
    :param param_vectors: List of flattened parameter vectors for the models.
    :return: A distance matrix.
    """
    # 计算参数向量之间的欧氏距离
    distance_matrix = squareform(pdist(param_vectors, metric='euclidean'))

    # 可视化相似度矩阵
    plt.figure(figsize=(8, 8))
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Model Parameter Distance Matrix')
    plt.show()
    return distance_matrix


# 可视化模型深度与参数差异的关系
def visualize_depth_vs_difference(models, param_vectors, tasks, distance_matrix):
    """
    Visualize the relationship between model depth and parameter difference.
    :param models: List of models.
    :param param_vectors: List of flattened parameter vectors for the models.
    :param tasks: List of corresponding tasks ('seg' or 'depth') for each model.
    :param distance_matrix: Precomputed distance matrix between models.
    """
    # 假设每个模型都有一个“深度”，可以通过模型结构的层数表示
    # 这里假设 get_model_depth 是一个函数，可以获取模型的深度
    depths = [get_model_depth(model) for model in models]  # 替换为实际获取模型深度的函数

    # 计算每个模型与其他模型的平均参数差异
    avg_param_diff_per_model = np.mean(distance_matrix, axis=1)

    # 可视化深度与参数差异的关系
    plt.figure(figsize=(8, 6))
    plt.scatter(depths, avg_param_diff_per_model, c=['red' if task == 'seg' else 'blue' for task in tasks])
    plt.title('Model Depth vs Parameter Difference')
    plt.xlabel('Model Depth')
    plt.ylabel('Average Parameter Difference')
    plt.legend(['Segmentation', 'Depth'])
    plt.show()
