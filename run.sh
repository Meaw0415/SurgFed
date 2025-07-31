
# Local Segmentation
# CUDA_VISIBLE_DEVICES=6 python train_3d.py -net sam2 -exp_name Endovis2017 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset Endovis2017 -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/Endovis2017/site1
# CUDA_VISIBLE_DEVICES=6 python train_3d.py -net sam2 -exp_name Endovis2018 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset Endovis2018 -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/Endovis2018/site2
# CUDA_VISIBLE_DEVICES=6 python train_3d.py -net sam2 -exp_name AutoLaparo_new -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset AutoLaparo -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/AutoLaparo
# # CUDA_VISIBLE_DEVICES=6 python train_3d.py -net sam2 -exp_name cholecseg8k -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset cholecseg8k -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/cholecseg8k


# # # Local Depth Estimation
# CUDA_VISIBLE_DEVICES=5 python train_3d_depth.py -net sam2 -exp_name SCARED -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset SCARED -data_path /mnt/iMVR/zhengf/MTFL/NYU_PAS_Dataset/SCARED
# CUDA_VISIBLE_DEVICES=6 python train_3d_depth.py -net sam2 -exp_name StereoMIS -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset StereoMIS -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/StereoMIS/StereoMIS

# FL SAM2

# CUDA_VISIBLE_DEVICES=2 python train_fl.py -net sam2 -exp_name FedAvg -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers both -num_nets 5
# CUDA_VISIBLE_DEVICES=6 python train_fl.py -net sam2 -exp_name FedAvg_task -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers both -num_nets 5 -Task_specific True
# CUDA_VISIBLE_DEVICES=7 python train_fl.py -net sam2 -exp_name FedRep -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers mem -num_nets 1
# CUDA_VISIBLE_DEVICES=4 python train_3d.py -net sam2 -exp_name Endovis2018 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset Endovis2018 -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/Endovis2018/site2

# Ours
# CUDA_VISIBLE_DEVICES=2 python train_hnfl.py -net sam2 -exp_name HNFL_both -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers both -num_nets 5 -local_epochs 3

CUDA_VISIBLE_DEVICES=1 python train_hnfl.py -net sam2 -exp_name HNFL_sam_1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers sam -num_nets 5 -local_epochs 1
CUDA_VISIBLE_DEVICES=5 python train_hnfl.py -net sam2 -exp_name SAM2_CLIP_1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers both -num_nets 5 -local_epochs 1
CUDA_VISIBLE_DEVICES=0 python train_fl.py -net sam2 -exp_name FedAvg -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers both -num_nets 5 -local_epochs 3
CUDA_VISIBLE_DEVICES=5 python train_fl.py -net sam2 -exp_name FedAvg_task -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers both -num_nets 5 -Task_specific True -local_epochs 3
# CUDA_VISIBLE_DEVICES=7 python train_fl.py -net sam2 -exp_name FedAvg_Decoder -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers sam -num_nets 5 -local_epochs 3
# CUDA_VISIBLE_DEVICES=6 python train_fl.py -net sam2 -exp_name FedRep -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers mem -num_nets 5 -local_epochs 1



CUDA_VISIBLE_DEVICES=1 python train_fl.py -net sam2 -exp_name nogl_sam -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers sam -num_nets 5 -local_epochs 3
CUDA_VISIBLE_DEVICES=2 python train_fl.py -net sam2 -exp_name nogl_mem -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers mem -num_nets 5 -local_epochs 3

CUDA_VISIBLE_DEVICES=5 python train_fedhca.py -net sam2 -exp_name SAM2_CLIP_1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -Layers both -num_nets 5 -local_epochs 1
# Vis 