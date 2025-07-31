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
    optimizer1s = []
    optimizer2s = []
    assert len(TASKS) == args.num_nets == len(DATASETS)

    
    for i in range(args.num_nets):
        if TASKS[i] == 'dep':
            net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed, if_depth=True, client_idx=i)
        else:
            net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed, if_depth=False, client_idx=i)
        net.to(dtype=torch.bfloat16)
        if args.pretrain:
            print(args.pretrain)
            weights = torch.load(args.pretrain)
            net.load_state_dict(weights,strict=False)
        nets.append(net)
    
        sam_layers = (
                        []
                    #   + list(net.image_encoder.parameters())
                    #   + list(net.sam_prompt_encoder.parameters())
                        + list(net.sam_mask_decoder.parameters())
                        )
        mem_layers = (
                        []
                        + list(net.obj_ptr_proj.parameters())
                        + list(net.memory_encoder.parameters())
                        + list(net.memory_attention.parameters())
                        + list(net.mask_downsample.parameters())
                        )
        
        if len(sam_layers) == 0:
            optimizer1 = None
        else:
            optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        if len(mem_layers) == 0:
            optimizer2 = None
        else:
            optimizer2 = optim.Adam(mem_layers, lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        optimizer1s.append(optimizer1)
        optimizer2s.append(optimizer2)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay
    
    # Init HyperCrossAttention
    hyper = HyperCrossAttention_embedding_res(nets, K=args.num_nets, init_beta=0.1, mode=args.Layers, site_texts=SITES_TEXT)
    hyper.to(dtype=torch.bfloat16)
    hyper.to(GPUdevice)
    hyper_optimizer = optim.Adam(hyper.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.99, amsgrad=False)


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

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    log_dir = os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
    writer = SummaryWriter(log_dir=log_dir)

    # checkpoint_ path is logdir/checkpoints
    checkpoint_path = os.path.join(args.path_helper['log_path'], 'checkpoints')
    visual_path = os.path.join(args.path_helper['log_path'], 'visual')
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    
    best_tol = 1e4
    best_dice_list = [0.0 for i in range(len(nets))]
    best_mse_list = [1e4 for i in range(len(nets))]
    weights_dict_list_not_update = []
    # weights_dict_list_updated = []
    last_param_dict_list_for_update = []
    # Init for hyper shall record  the dict of last param after Aggregation
    last_param_dict_list = []
    last_ckpts = []
    for i in range(args.num_nets):
        weights_dict_list_not_update.append(nets[i].state_dict())
        # weights_dict_list_updated.append(nets[i].state_dict())
        
        last_param_dict_list.append(nets[i].state_dict())
    last_ckpts = [copy.deepcopy(nets[i].state_dict()) for i in range(args.num_nets)]
    
    for com_round in range(args.com_rounds):
        print("Communication round ", com_round)
        time_com_start = time.time()
        for i in range(args.num_nets):
            current_net = nets[i]
            current_optimizer1 = optimizer1s[i]
            current_optimizer2 = optimizer2s[i]
            current_nice_train_loader = nice_train_loaders[i]
            current_nice_test_loader = nice_test_loaders[i]
            current_dataset = DATASETS[i]
            current_task = TASKS[i]

            for local_epoch in range(args.local_epochs):
                current_net.train()
                time_start = time.time()
                if current_task == 'dep':
                    loss, prompt_loss, non_prompt_loss = function_depth.train_sam(args, current_net, current_optimizer1, current_optimizer2, current_nice_train_loader, local_epoch)
                else:
                    loss, prompt_loss, non_prompt_loss = function.train_sam(args, current_net, current_optimizer1, current_optimizer2, current_nice_train_loader, local_epoch)
                logger.info(f'Dataset: {current_dataset} || Task: {current_task} || Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {local_epoch}.')
                time_end = time.time()
                print(f'Dataset: {current_dataset} || Task: {current_task} || Time for training: {time_end - time_start} seconds')

            # Update weights
            weights_dict_list_not_update[i] = current_net.state_dict()
            # Evaluation
            current_net.eval()  
            if current_task == 'dep':
                tol, (eiou, edice) = function_depth.validation_sam(args, current_nice_test_loader, com_round, current_net, visual_path, writer)
                logger.info(f'Dataset: {current_dataset} || Task: {current_task} || Total score: {tol}, MAE: {eiou}, RMSE: {edice} || @ Communication rounds {com_round}.')
            else:
                tol, (eiou, edice) = function.validation_sam(args, current_nice_test_loader, com_round, current_net, visual_path, writer)
                logger.info(f'Dataset: {current_dataset} || Task: {current_task} || Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ Communication rounds {com_round}.')
            
            if current_task == 'dep':
                if eiou < best_mse_list[i]:
                    best_mse_list[i] = eiou
                    # vis the depth map
                    # function_depth.vis_sam(args, current_nice_test_loader, com_round, current_net, visual_path, writer,vis=True)
                    torch.save(current_net.state_dict(), checkpoint_path.format(net=i, epoch=com_round, type='best_mse'))
            else:
                if edice > best_dice_list[i]:
                    best_dice_list[i] = edice
                    # function.vis_sam(args, current_nice_test_loader, com_round, current_net, visual_path, writer,vis=True)
                    torch.save(current_net.state_dict(), checkpoint_path.format(net=i, epoch=com_round, type='best_dice'))
                

        # # Aggregation and update
        # weights_dict_list_updated = FedAvg(weights_dict_list_not_update, num_clients = args.num_nets, Tasks=TASKS,  Layers=args.Layers, client_weights=WEIGHTS_CLIENTS, Task_specific=args.Task_specific)
        # weights_dict_list_updated = MaTFL(weights_dict_list_not_update, Tasks = TASKS, num_clients = args.num_nets, Layers=args.Layers, Task_specific=args.Task_specific)
        if com_round > 0:
            HyperCrossAttention_Update(weights_dict_list_not_update, last_param_dict_list, args.num_nets, hyper_optimizer=hyper_optimizer,hypernetwork=hyper, last_ckpts=last_ckpts), 
        
        last_param_dict_list = Hyper_Aggregate(weights_dict_list_not_update, last_param_dict_list, args.num_nets, hyper_optimizer=hyper_optimizer,hypernetwork=hyper, last_ckpts=last_ckpts)

        for i in range(args.num_nets):
            nets[i].load_state_dict(last_param_dict_list[i])
            last_ckpts[i] = copy.deepcopy(nets[i].state_dict())

            
        
        time_com_end = time.time()
        print("Time for communication round ", com_round, ": ", time_com_end - time_com_start, " seconds")




    writer.close()


if __name__ == '__main__':
    main()