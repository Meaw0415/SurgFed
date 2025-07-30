""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm
import cv2
import cfg
from conf import settings
from func_3d.utils import eval_seg

args = cfg.parse_args()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal

# class DepthLoss(nn.Module):
#     """
#     Loss for depth prediction. By default L1 loss is used.  
#     """
#     def __init__(self, loss='l1'):
#         super(DepthLoss, self).__init__()
#         if loss == 'l1':
#             self.loss = nn.L1Loss()

#         else:
#             raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))

#     def forward(self, out, label):
#         mask = (label != 255)
#         return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))
    
class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """
    def __init__(self, loss='l1'):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))

    def forward(self, out, label):
        # Create mask for valid pixels (where label is not 255)
        mask = (label != 255)
        
        # Create an additional mask where label is zero
        zero_mask = (label == 0)
        
        # Set `out` values to zero where `label` is zero
        out = out * ~zero_mask  # Invert zero_mask to apply where `label` is not zero
        
        # Calculate L1 loss only on valid pixels where label is not 255
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class ProxLoss(nn.Module):
    def __init__(self, mu=1):
        """
        """
        super(ProxLoss, self).__init__()
        self.mu = mu

    def forward(self, local_params, global_params):
        """
        """
        prox_loss = 0.0
        for local_param, global_param in zip(local_params, global_params):
            if local_param.requires_grad:
                prox_loss += torch.sum((local_param - global_param.detach()) ** 2)
        return self.mu / 2 * prox_loss
    


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Loss function for the depth prediction

func_dep_loss = DepthLoss(loss='l1')
paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
prox_loss = ProxLoss(mu=1)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []



def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader,
          epoch, global_model= None):
    hard = 0
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    video_length = args.video_length

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss#.to(dtype=torch.bfloat16, device=GPUdevice)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            depth_tensor = pack['depths']

            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # reverse = np.random.rand() > 0.5

            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ , _= net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                depth_maps_dict = {}
                for out_frame_idx, out_obj_ids, out_mask_logits, depth_maps in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    depth_maps_dict[out_frame_idx] = {
                        out_obj_id: depth_maps[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        depth_map = depth_maps_dict[id][ann_obj_id]
                        depth_map = depth_map.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.train_vis:
                            os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            try:
                                bbox = bbox_dict[id][ann_obj_id]
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        
                        # depth tensor shape: batch, video_length, obj_num, h w
                        # depth map shape: batch, objnum, h, w
                        obj_loss = func_dep_loss(depth_map, depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).to(dtype = torch.float32, device = GPUdevice))
                        # obj_loss = lossfunc(pred, mask)
                        loss += obj_loss.item()
                        if global_model is not None:
                            loss += prox_loss(net.state_dict().values(), global_model.values())
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss.backward()
                    optimizer1.step()
                
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()
                net.reset_state(train_state)

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, vis_path ,clean_dir=True, vis=False):
     # eval mode
    
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            depth_tensor = pack['depths']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ , _= net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                depth_maps_dict = {}
                for out_frame_idx, out_obj_ids, out_mask_logits, depth_maps in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                    depth_maps_dict[out_frame_idx] = {
                        out_obj_id: depth_maps
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                pred_RMSE = 0
                pred_MAE = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        depth_map = depth_maps_dict[id][ann_obj_id]
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if vis:
                            # os.makedirs(f'./temp/val/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 2)
                            ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].cpu().numpy())
                            ax[1].axis('off')
                            # ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                            # ax[2].axis('off')
                        
                            # save fig to vis_path/name[0]/id/ann_obj_id.png
                            final_vis_path = os.path.join(vis_path, name[0], str(id))
                            os.makedirs(final_vis_path, exist_ok=True)
                            plt.savefig(os.path.join(final_vis_path, f'{ann_obj_id}.png'), bbox_inches='tight', pad_inches=0)

                            # plt.savefig(f'./temp/val/{name[0]}/{id}/{ann_obj_id}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        # loss += lossfunc(pred, mask)
                        loss += func_dep_loss(depth_map, depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).to(dtype = torch.float32, device = GPUdevice)).item()
                        # Eval RMSE between depth map and ground truth
                        
                        # Normalize depth_map by x-x.min()/x.max()-x.min() * 255
                        depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                       
                        # same for depth_tensor
                        depth_tensor_norm = (depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=GPUdevice) - depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).min()) / (depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).max() - depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).min()) * 255 

                        # Create a mask for valid pixels where depth_tensor is non-zero
                        

                        # Calculate RMSE only on valid pixels
                        pred_RMSE += torch.sqrt(torch.mean((depth_map_norm - depth_tensor_norm)**2)).item()
                        pred_MAE += torch.mean(torch.abs(depth_map_norm - depth_tensor_norm)).item()
                        # pred_RMSE += torch.sqrt(torch.mean((depth_map/255 - depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).to(dtype = torch.float32, device = GPUdevice))**2)).item()

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_MAE / total_num, pred_RMSE / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])

def vis_sam(args, val_loader, epoch, net: nn.Module, vis_path ,clean_dir=True, vis=False):
     # eval mode
    
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            depth_tensor = pack['depths']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ , _= net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                depth_maps_dict = {}
                for out_frame_idx, out_obj_ids, out_mask_logits, depth_maps in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                    depth_maps_dict[out_frame_idx] = {
                        out_obj_id: depth_maps
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                pred_RMSE = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        depth_map = depth_maps_dict[id][ann_obj_id]
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if vis:
                            fig, ax = plt.subplots(1, 3)  # 增加一个 subplot
                            ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].set_title("Input Image")
                            ax[0].axis('off')
                            
                        
                            # Normalize by x-xmin
                            depth_map_norm = (depth_map-depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                            depth_map_norm = depth_map_norm.to(dtype=torch.uint8)

                            # depth_map_norm = quantize_to_given_values(depth_map_norm)
                            depth_map_norm = quantize_to_n_values(depth_map_norm,num_intervals=8)
                            quantize_to_n_values
                            # 显示处理后的预测深度图
                            ax[1].imshow(depth_map_norm.squeeze(0).squeeze(0).cpu().numpy(), cmap="grey", interpolation="bilinear")
                            ax[1].set_title("Predicted Depth Map")
                            ax[1].axis("off")

                            depth_tensor_norm = depth_tensor[0, id, obj_list.index(ann_obj_id)]
                            depth_tensor_norm = (depth_tensor_norm - depth_tensor_norm.min()) / (depth_tensor_norm.max() - depth_tensor_norm.min()) * 255

                            # 转换为 uint8
                            depth_tensor_norm = depth_tensor_norm.to(dtype=torch.uint8)

                            # 直方图均衡化
                            # depth_tensor_norm = histogram_equalization(depth_tensor_norm)

                            # 量化到 20 级
                            # depth_tensor_norm = quantize_to_given_values(depth_tensor_norm)

                            # 显示处理后的 GT 深度图
                            ax[2].imshow(depth_tensor_norm.cpu().numpy(), cmap="grey", interpolation="bilinear")
                            ax[2].set_title("GT Depth Map")
                            ax[2].axis("off")

                            # 计算MSE 并放在标题上
                            # print(depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).cpu())
                            valid_mask = (depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=GPUdevice) != 0)

                            # Set depth_map values to zero where depth_tensor is zero
                            depth_map_norm = depth_map * valid_mask
                            rmse =  torch.sqrt(torch.mean(((depth_map_norm - depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=GPUdevice)) ** 2)[valid_mask])).item()
                            
                            # 计算 MAE
                            mae = torch.mean(((depth_map_norm - depth_tensor[0, id, obj_list.index(ann_obj_id), :, :].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=GPUdevice)) )[valid_mask]).item()

                            ax[2].set_title(f"GT Depth Map (MSE: {rmse:.8f}) MAE: {mae:.4f}")

                            # 保存图像
                            final_vis_path = os.path.join(vis_path, name[0], str(id))
                            os.makedirs(final_vis_path, exist_ok=True)
                            plt.savefig(os.path.join(final_vis_path, f'{ann_obj_id}.png'), bbox_inches='tight', pad_inches=0)
                            plt.close()

    return 




def quantize_to_given_values(depth_map, target_values=[30,40,50,60,90,100,110,120]):
    """
    将 depth_map 的像素值映射到给定的深度值列表中最接近的值，实现自定义分层量化。

    :param depth_map: torch.Tensor，归一化后的深度图 (0-255)，shape: (N, 1, H, W)
    :param target_values: list or torch.Tensor，自定义深度值列表，例如 [10, 20, 30, 40]
    :return: 量化后的深度图 (torch.uint8)，形状与输入相同
    """
    depth_map = depth_map.clone().to(torch.float32)  # 确保为 float 进行计算
    target_values = torch.tensor(target_values, device=depth_map.device, dtype=torch.float32).view(-1, 1, 1, 1)

    # 计算每个像素与目标深度值的差异
    diff = torch.abs(depth_map - target_values)  # shape: (num_values, N, H, W)

    # 找到差异最小的索引
    closest_idx = torch.argmin(diff, dim=0)  # shape: (N, H, W)

    # 根据索引将像素映射到对应的目标深度值
    quantized_map = target_values[closest_idx, 0, 0, 0].to(torch.uint8)

    return quantized_map




def quantize_to_n_values(depth_map, num_intervals=5, min_val=0, max_val=255):
    """
    让 depth_map 只包含 num_intervals 个等间隔值，排除 0 和 255，确保平滑不模糊。

    输入:
    - depth_map: torch.Tensor，归一化后的深度图 (0-255)
    - num_intervals: int，量化的区间数
    - min_val: int，最小值 (默认 0)
    - max_val: int，最大值 (默认 255)

    输出:
    - quantized_map: 量化后的深度图，torch.uint8 类型
    """
    depth_map = depth_map.clone()
    quantized_map = torch.zeros_like(depth_map, dtype=torch.uint8, device=depth_map.device)

    # 生成不包含 0 和 255 的等间隔值
    target_values = torch.linspace(min_val, max_val, num_intervals, device=depth_map.device).to(torch.uint8)

    # 遍历 target_values，找到每个像素最近的目标值
    for val in target_values:
        val = val.to(torch.uint8)
        mask = (torch.abs(depth_map - val) < torch.abs(depth_map - quantized_map))
        quantized_map[mask] = val

    return quantized_map





def floodfill_binary_layer(layer_binary):
    """
    对二值图进行 8 邻域 FloodFill 填洞。

    输入:
    - layer_binary: numpy.ndarray，二值图 (H, W)，像素值为 0 或 1。

    输出:
    - filled_layer: numpy.ndarray，填补空洞后的二值图 (H, W)。
    """
    h, w = layer_binary.shape
    layer_uint8 = (layer_binary * 255).astype(np.uint8)  # 转换为 0/255 图像

    # 创建 FloodFill 所需掩码
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(layer_uint8, mask, (0, 0), 255, flags=8)  # 使用 8 邻域填充

    # 反转填充图以获得空洞区域
    inverted_flood_filled = cv2.bitwise_not(layer_uint8)
    filled_layer = (layer_uint8 | inverted_flood_filled) // 255  # 恢复为 0/1

    return filled_layer


def fill_depthmap_with_floodfill(depth_map_tensor, num_intervals=5):
    """
    基于量化 + 二值 FloodFill 分层填补深度图空洞。

    输入:
    - depth_map_tensor: (1, 1, H, W) 的深度图张量。
    - num_intervals: 量化深度的区间数。

    输出:
    - filled_depth_map: 填补后的深度图，形状与输入相同。
    """
    # Step 1: 归一化 & 量化
    depth_map = depth_map_tensor.squeeze(0).squeeze(0)
    min_val, max_val = depth_map.min(), depth_map.max()
    depth_map_norm = ((depth_map - min_val) / (max_val - min_val) * 255).to(torch.uint8)
    quantized_map = quantize_to_n_values(depth_map_norm, num_intervals=num_intervals)

    # Step 2: 逐层 FloodFill 填补
    target_values = torch.unique(quantized_map).sort()[0]  # 获取有序唯一深度值
    final_filled = torch.zeros_like(quantized_map, dtype=torch.uint8)
    final_filled = final_filled.cpu().numpy()
    # Transfer target_values to numpy
    target_values = target_values.cpu().numpy()
    for val in reversed(target_values):
        if val == 0 or val == 255:
            continue
        # 创建二值图：当前深度值为1，其余为0
        layer_binary = (quantized_map == val).cpu().numpy().astype(np.uint8)

        # 对二值图进行 FloodFill
        filled_binary = floodfill_binary_layer(layer_binary)

        # 填补完成后乘以当前深度值
        filled_layer = filled_binary * val

        # 高深度值覆盖低深度值
        final_filled[filled_layer > 0] = filled_layer[filled_layer > 0]

        
    print(np.unique(final_filled))
    # Step 4: 反归一化
    filled_depth_map = torch.from_numpy(final_filled).to(torch.float32).to(depth_map_tensor.device)

    filled_depth_map = filled_depth_map / 255 * (max_val - min_val) + min_val

    return filled_depth_map.unsqueeze(0).unsqueeze(0)





