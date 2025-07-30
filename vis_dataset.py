#!/usr/bin/env python3

""" train network using pytorch
    Yunli Qi
"""

import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from scipy.stats import entropy, ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from torchmetrics.image.fid import FrechetInceptionDistance

import cfg
from func_3d import function, function_depth
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from Aggregate import FedAvg, MaTFL

# 设置路径
VIS_RESULTS_PATH = "/mnt/iMVR/zhengf/Medical-SAM2/vis_results"

# 任务和数据集定义
TASKS = [
    'dep',
    'dep',
    'seg',
    'seg',
    'seg',
]
DATASETS = [
    'StereoMIS',
    'EndoNerf',
    'Endovis2017',
    'Endovis2018',
    'AutoLaparo',
]
WEIGHTS_CLIENTS = [1, 1, 8, 15, 170]

# 提取数据集特征
def extract_features(dataloader):
    features = []
    for data in dataloader:
        inputs = data['image']  # 假设输入是特征
        features.append(inputs.view(len(inputs), -1).cpu().numpy())  # 将样本展平成2D
    return np.concatenate(features, axis=0)

# t-SNE 可视化并保存
def tsne_visualization(features_list, labels, dim=2, save_path=None):
    tsne = TSNE(n_components=dim, random_state=42)
    features_tsne = tsne.fit_transform(np.concatenate(features_list, axis=0))
    
    # 为每个数据集分配不同颜色
    colors = ['r', 'g', 'b', 'c', 'm']  # 根据数据集数量来设置不同颜色

    plt.figure()
    start_idx = 0
    for i, features in enumerate(features_list):
        end_idx = start_idx + len(features)
        if dim == 2:
            plt.scatter(features_tsne[start_idx:end_idx, 0], features_tsne[start_idx:end_idx, 1], 
                        c=colors[i], label=labels[i])
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(features_tsne[start_idx:end_idx, 0], features_tsne[start_idx:end_idx, 1], 
                       features_tsne[start_idx:end_idx, 2], c=colors[i], label=labels[i])
        start_idx = end_idx
    
    plt.legend()
    plt.title(f'{dim}D t-SNE Visualization')
    if save_path:
        plt.savefig(save_path)
    plt.close()

# 计算 JS 散度
def js_divergence(p, q):
    return jensenshannon(p, q)

# 计算 Wasserstein 距离
def wasserstein_distance_matrix(datasets):
    dist_matrix = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            dist_matrix[i, j] = wasserstein_distance(datasets[i].ravel(), datasets[j].ravel())
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

# 计算 Kolmogorov-Smirnov 检验
def ks_test_matrix(datasets):
    ks_matrix = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            stat, p_value = ks_2samp(datasets[i].ravel(), datasets[j].ravel())
            ks_matrix[i, j] = stat
            ks_matrix[j, i] = stat
    return ks_matrix

# 计算 FID
fid_metric = FrechetInceptionDistance()

def calculate_fid_matrix(datasets):
    fid_metric = FrechetInceptionDistance(feature=64)  # 确保选择合适的feature层
    fid_matrix = np.zeros((len(datasets), len(datasets)))

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            # 将每个数据集的图像从 uint8 转换为 float32 并归一化到 [0, 1]
            dataset_i = torch.tensor(datasets[i], dtype=torch.float32) / 255.0
            dataset_j = torch.tensor(datasets[j], dtype=torch.float32) / 255.0

            # 如果数据不是 [batch_size, num_channels, height, width] 格式，需要调整维度
            if len(dataset_i.shape) == 3:  # 假设数据是 [batch_size, height, width]
                dataset_i = dataset_i.unsqueeze(1)  # 添加 channel 维度
            if len(dataset_j.shape) == 3:
                dataset_j = dataset_j.unsqueeze(1)

            # 确保通道数是 3，如果是单通道则重复三次以模拟 RGB
            if dataset_i.shape[1] == 1:
                dataset_i = dataset_i.repeat(1, 3, 1, 1)
            if dataset_j.shape[1] == 1:
                dataset_j = dataset_j.repeat(1, 3, 1, 1)

            # 使用 FID 度量
            fid_metric.update(dataset_i, real=True)
            fid_metric.update(dataset_j, real=False)
            fid_matrix[i, j] = fid_metric.compute()
            fid_matrix[j, i] = fid_matrix[i, j]

            # 重置 FID 度量，以确保下一次更新不受干扰
            fid_metric.reset()

    return fid_matrix

def visualize_heatmap(matrix, title, save_path, datasets):
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    
    # 添加颜色条
    plt.colorbar()
    
    # 设置标题
    plt.title(title)
    
    # 设置坐标为数据集名称
    plt.xticks(np.arange(len(datasets)), datasets, rotation=45, ha="right")
    plt.yticks(np.arange(len(datasets)), datasets)
    
    # 在每个格子中显示具体数值
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
    
    # 调整布局以避免标签被裁剪
    plt.tight_layout()
    
    # 保存热力图
    plt.savefig(save_path)
    plt.close()


def main():
    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    nice_train_loaders = []
    nice_test_loaders = []

    # 提取各个数据集的 DataLoader
    for i in range(args.num_nets):
        args.dataset = DATASETS[i]
        nice_train_loader, nice_test_loader = get_dataloader(args)
        nice_train_loaders.append(nice_train_loader)
        nice_test_loaders.append(nice_test_loader)
        print("Dataset: ", DATASETS[i], " || Train Clips: ", len(nice_train_loader), " || Test Clips: ", len(nice_test_loader))

    # 提取每个数据集的特征
    all_train_features = [extract_features(loader) for loader in nice_train_loaders]

    # 2D t-SNE 可视化并保存
    # tsne_2d_save_path = os.path.join(VIS_RESULTS_PATH, 'tsne_2d.png')
    # tsne_visualization(all_train_features, DATASETS, dim=2, save_path=tsne_2d_save_path)
    # print(f"2D t-SNE visualization saved at {tsne_2d_save_path}")

    # 3D t-SNE 可视化并保存
    # tsne_3d_save_path = os.path.join(VIS_RESULTS_PATH, 'tsne_3d.png')
    # tsne_visualization(all_train_features, DATASETS, dim=3, save_path=tsne_3d_save_path)
    # print(f"3D t-SNE visualization saved at {tsne_3d_save_path}")

    # 计算分布差异
    wasserstein_distances = wasserstein_distance_matrix(all_train_features)
    print("Wasserstein distances:\n", wasserstein_distances)
    ks_distances = ks_test_matrix(all_train_features)
    print("KS-test distances:\n", ks_distances)

    visualize_heatmap(wasserstein_distances, 'Wasserstein Distances', os.path.join(VIS_RESULTS_PATH, 'wasserstein_distances.png'), DATASETS)
    visualize_heatmap(ks_distances, 'KS-test Distances', os.path.join(VIS_RESULTS_PATH, 'ks_test_distances.png'), DATASETS)

    
    
    

  

   


if __name__ == '__main__':
    main()
