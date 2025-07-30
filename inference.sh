
# Local Segmentation
CUDA_VISIBLE_DEVICES=1 python train_3d.py -net sam2 -exp_name Endovis2017 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset Endovis2017 -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/Endovis2017/site1
CUDA_VISIBLE_DEVICES=1 python train_3d.py -net sam2 -exp_name Endovis2018 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset Endovis2018 -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/Endovis2018/site2
CUDA_VISIBLE_DEVICES=1 python train_3d.py -net sam2 -exp_name AutoLaparo_new -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset AutoLaparo -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/AutoLaparo
CUDA_VISIBLE_DEVICES=1 python train_3d.py -net sam2 -exp_name cholecseg8k -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset cholecseg8k -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/cholecseg8k


# Local Depth Estimation
CUDA_VISIBLE_DEVICES=0 python train_3d_depth.py -net sam2 -exp_name EndoNerf -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset EndoNerf -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/endonerf_sample_datasets
CUDA_VISIBLE_DEVICES=0 python train_3d_depth.py -net sam2 -exp_name EndoNerf -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_dep -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset StereoMIS -data_path /mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/StereoMIS/StereoMIS