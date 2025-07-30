# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time

import copy
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d import function
from func_3d import function_depth
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from func_3d.hyper import HyperCrossAttention, HyperCrossAttention_embedding_res
from Aggregate import FedAvg, MaTFL, HyperCrossAttention_Update, Hyper_Aggregate

TASKS    = [
            'dep',
            'dep',
            'seg',
            'seg',
            'seg',

            # 'seg',
            ]

DATASETS = [
            'StereoMIS',
            'SCARED',
            'Endovis2017',
            'Endovis2018',
            'AutoLaparo',
            # 'cholecseg8k'
            ] 

WEIGHTS_CLIENTS = [
                    1,
                    1,
                    8,
                    15,
                    170,
                    # 76,
                    ]

SITES_TEXT = [  
                'Dataset: StereoMIS. Task: Depth Estimation. Label: The distance of each pixel relative to the camera ',
                'Dataset: SCARED. Task: Depth Estimation. Lbael: The distance of each pixel relative to the camera ',
                'Dataset: Endovis2017. Task: Instrument Segmentation. Label: Shaft, Wrist, Clasper ',
                'Dataset: Endovis2018. Task: Scene Segmentation. Label:Shaft,Wrist,Clasper,kidney-parenchyma,covered-kidney,hread,clamps,suturing-needle,suction-instrument,small-intestine,ultrasound-probe ',
                'Dataset: AutoLaparo. Task: Instrument Segmentation. Label: Anatomy Uterus , Shaft and Manipulator of Grasping forceps, LigaSure, Dissecting grasping forceps and Electric hook',

                # 'Dataset: cholecseg8k. Task: seg. ',
]

def main():

    args = cfg.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)
    nets = []
    args.num_nets = 5
    assert len(TASKS) == args.num_nets == len(DATASETS)

    
    for i in range(args.num_nets):
        if TASKS[i] == 'dep':
            net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed, if_depth=True, client_idx=i)
        else:
            net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed, if_depth=False, client_idx=i)
        net.to(dtype=torch.bfloat16)
        nets.append(net)

    # Load the pre-trained model
    # path_dir = '/mnt/iMVR/zhengf/Medical-SAM2/logs/FedProx_2025_01_14_16_29_44/Log/checkpoints'
    # path_dir = '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/MaTFL_2024_12_18_15_17_34/Log/checkpoints'
    # path_dir = '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/FedRep_2024_12_18_22_25_48/Log/checkpoints'
    # path_dir = '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/FedHCA_2024_12_18_22_05_23/Log/checkpoints'
    # path_dir = '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/FedAvg_2024_12_18_14_58_55/Log/checkpoints'
    # path_dir = '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/ours_2025_01_30_00_26_08/Log/checkpoints'
    # path_dir = '/mnt/iMVR/zhengf/Medical-SAM2/logs/HNFL_both_2025_02_20_21_39_06/Log/checkpoints copy'

    path_dir_list = [
        '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/FedAvg_2024_12_18_14_58_55/Log/checkpoints',
    #     '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/FedRep_2024_12_18_22_25_48/Log/checkpoints',
    #    '/mnt/iMVR/zhengf/Medical-SAM2/logs/FedProx_2025_01_14_16_29_44/Log/checkpoints',
    #    '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/MaTFL_2024_12_18_15_17_34/Log/checkpoints',
    #     '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/FedHCA_2024_12_18_22_05_23/Log/checkpoints',
    #     '/mnt/iMVR/zhengf/Medical-SAM2/logs/vis/ours_2025_01_30_00_26_08/Log/checkpoints'
    ]

    visual_path_list = [
        '/mnt/iMVR/zhengf/Medical-SAM2/vis/seg1/FedAvg',
        # '/mnt/iMVR/zhengf/Medical-SAM2/vis/seg1/FedRep',
        # '/mnt/iMVR/zhengf/Medical-SAM2/vis/seg1/FedProx',
        # '/mnt/iMVR/zhengf/Medical-SAM2/vis/seg1/MaTFL',
        # '/mnt/iMVR/zhengf/Medical-SAM2/vis/seg1/FedHCA',
        # '/mnt/iMVR/zhengf/Medical-SAM2/vis/seg1/ours'
    ]

    for idx,path_dir in enumerate(path_dir_list):
        visual_path = visual_path_list[idx]

        path_list = []
        # add pth file into list the file should be in the path_dir and rank them by name
        for file in os.listdir(path_dir):
            if file.endswith(".pth"):
                path_list.append(os.path.join(path_dir, file))
        path_list.sort()

        for i in range(args.num_nets):
            checkpoint = torch.load(path_list[i])
            nets[i].load_state_dict(checkpoint,strict = False)
            print("Load the pre-trained model from: ", path_list[i])
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)

        nice_train_loaders = []
        nice_test_loaders = []

        for i in range(args.num_nets):
            args.dataset = DATASETS[i]
            nice_train_loader, nice_test_loader = get_dataloader(args)
            nice_train_loaders.append(nice_train_loader)
            nice_test_loaders.append(nice_test_loader)
            print("Dataset: ", DATASETS[i], " || Train Clips: ", len(nice_train_loader), " || Test Clips: ", len(nice_test_loader))

        
        
        weights_dict_list_not_update = []

        # Init for hyper shall record  the dict of last param after Aggregation
        last_param_dict_list = []

        for i in range(args.num_nets):
            weights_dict_list_not_update.append(nets[i].state_dict())
            # weights_dict_list_updated.append(nets[i].state_dict())
            last_param_dict_list.append(nets[i].state_dict())
        
        
        for i in range(args.num_nets):
            current_net = nets[i]

            current_nice_test_loader = nice_test_loaders[i]
            current_nice_train_loader = nice_train_loaders[i]
            current_task = TASKS[i]

            # Update weights
            weights_dict_list_not_update[i] = current_net.state_dict()
            # Evaluation
            current_net.eval()  
            
            if current_task == 'dep':
                continue
                tol, (eiou, edice) = function_depth.validation_sam(args, current_nice_test_loader, 0, current_net, visual_path, None)

                print("Dataset: ", DATASETS[i], " || Task: ", current_task, " || MAE: ", eiou, " || RMSE: ", edice)
                function_depth.vis_sam(args, current_nice_test_loader, 0, current_net, visual_path, None,vis=True)
                function_depth.vis_sam(args, current_nice_train_loader, 0, current_net, visual_path, None,vis=True)
                # continue
            else:
                # continue
                function.vis_sam(args, current_nice_test_loader, 0, current_net, visual_path, None,vis=False)

                



if __name__ == '__main__':
    main()