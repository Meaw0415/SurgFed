""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from func_3d.utils import eval_seg

args = cfg.parse_args()

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


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Loss function for the depth prediction

paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
seed = torch.randint(1,11,(1,7))
prox_loss = ProxLoss(mu=1)

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []



def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader,
          epoch,global_model=None):
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
            # depth_tensor = pack['depths']

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
                                _, _, _ = net.train_add_new_bbox(
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
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        
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
                        
                        obj_loss = lossfunc(pred, mask)
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
                                _, _, _ = net.train_add_new_bbox(
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
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if vis:
                            # os.makedirs(f'./temp/val/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            final_vis_path = os.path.join(vis_path, name[0], str(id))
                            os.makedirs(final_vis_path, exist_ok=True)
                            plt.savefig(os.path.join(final_vis_path, f'{ann_obj_id}.png'), bbox_inches='tight', pad_inches=0)

                            plt.close()
                        loss += lossfunc(pred, mask)
                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        pred_dice += temp[1]

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])

def vis_sam(args, val_loader, epoch, net: nn.Module, vis_path, clean_dir=True, vis=False):
    # eval mode
    net.eval()
    alpha = 0.8
    n_val = len(val_loader)  # the number of batches
    prompt_freq = args.prompt_freq
    prompt = args.prompt

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']

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
                                net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )

                video_segments = {}  # Contains per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }



                for id in frame_id:
                    if name[0] not in ['video4', 'video10', '258']:
                        continue
                    if id not in [27, 72, 3]:
                        continue

                    # 🎯 创建单通道 mask，初始为 0 表示背景
                    pred_combined_mask = np.zeros(imgs_tensor.shape[2:], dtype=np.uint8)
                    label_combined_mask = np.zeros(imgs_tensor.shape[2:], dtype=np.uint8)

                    # ✅ 遍历每个对象并生成类别 ID mask
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id].unsqueeze(0)
                        pred_mask = (pred[0, 0, :, :].cpu().numpy() > 0.5).astype(np.uint8)

                        try:
                            label_mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                            label_mask = label_mask[0, 0, :, :].cpu().numpy().astype(np.uint8)
                        except KeyError:
                            label_mask = np.zeros_like(pred_mask, dtype=np.uint8)

                        # 🎨 对预测和标签赋值类别 ID（覆盖模式）
                        pred_combined_mask[pred_mask == 1] = int(ann_obj_id)
                        label_combined_mask[label_mask == 1] = int(ann_obj_id)

                    # 💾 保存为单通道图像
                    save_path = os.path.join(vis_path, name[0], str(id))
                    os.makedirs(save_path, exist_ok=True)
                    plt.imsave(os.path.join(save_path, 'pred_mask_classid.png'), pred_combined_mask, cmap='gray')
                    plt.imsave(os.path.join(save_path, 'label_mask_classid.png'), label_combined_mask, cmap='gray')
                    print("done")
                    # print(f"✅ 已保存: {os.path.join(save_path, 'pred_mask_classid.png')} 和 label_mask_classid.png")

                # for id in frame_id:
                #     if name[0] not in ['video4', 'video10', '258']:
                #         continue
                #     if id not in [27,72,3]:
                #         continue
                #     pred_combined_mask = np.zeros((*imgs_tensor.shape[2:], 3), dtype=np.float32)  # Three-channel mask
                #     label_combined_mask = np.zeros((*imgs_tensor.shape[2:], 3), dtype=np.float32)


                #     img = imgs_tensor[id].cpu().permute(1, 2, 0).numpy().astype(np.float32) / 255.0
                #     # save img
                #     os.makedirs(os.path.join(vis_path, name[0], str(id)), exist_ok=True)
                #     plt.imsave(os.path.join(vis_path, name[0], str(id), 'img.png'), img)
                    
                #     for ann_obj_id in obj_list:
                #         pred = video_segments[id][ann_obj_id].unsqueeze(0)
                #         pred_mask = (pred[0, 0, :, :].cpu().numpy() > 0.5).astype(np.float32)
                #         try:
                #             label_mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                #             label_mask = label_mask[0, 0, :, :].cpu().numpy().astype(np.float32)
                #         except KeyError:
                #             label_mask = np.zeros_like(pred_mask, dtype=np.float32)

                #         unique_colors = plt.cm.tab20.colors
                #         color = unique_colors[int(ann_obj_id)]

                #         for c in range(3):  # Apply color to each channel
                            
                #             # pred_combined_mask[:, :, c] += pred_mask * color[c]
                #             pred_combined_mask[pred_mask == 1, c] = color[c]
                #             # pred_combined_mask[pred_mask == 1] = color[c]
                #             label_combined_mask[:, :, c] += label_mask * color[c]

                    if vis:
                        final_vis_path = os.path.join(vis_path, name[0], str(id))
                        os.makedirs(final_vis_path, exist_ok=True)
                        
                        alpha = 0.3
                        # 确保 pred_combined_mask 和 label_combined_mask 归一化到 0-1
                        pred_combined_mask = np.clip(pred_combined_mask, 0, 1)
                        label_combined_mask = np.clip(label_combined_mask, 0, 1)

                        # 更标准的 overlay 计算方法
                        pred_overlay = img * (1 - pred_combined_mask * alpha) + pred_combined_mask
                        label_overlay = img * (1 - label_combined_mask * alpha) + label_combined_mask

                        # Save overlays
                        fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=800)
                        ax[0].imshow(pred_overlay)
                        ax[0].axis('off')

                        ax[1].imshow(label_overlay)
                        ax[1].axis('off')

                        plt.savefig(os.path.join(final_vis_path, f'overlay.png'), bbox_inches='tight', pad_inches=0)
                        plt.close()
                        print("save overlay")

                        
                        # # Create overlays with alpha blending for pred and label masks
                        # pred_overlay = (1 - alpha) * img + pred_combined_mask * alpha
                        # label_overlay = (1 - alpha) * img + label_combined_mask * alpha

                        # # Save overlays
                        # fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=800)
                        # ax[0].imshow(pred_overlay)
                        # ax[0].axis('off')

                        # ax[1].imshow(label_overlay)
                        # ax[1].axis('off')

                        # plt.savefig(os.path.join(final_vis_path, f'overlay.png'), bbox_inches='tight', pad_inches=0)
                        # plt.close()
                        # print("save overlay")
                        # # Save pred mask
                        # fig, ax = plt.subplots()
                        # ax.imshow(pred_combined_mask / pred_combined_mask.max())  # Normalize for visualization
                        # ax.axis('off')
                        # plt.savefig(os.path.join(final_vis_path, f'pred_mask.png'), bbox_inches='tight', pad_inches=0)
                        # plt.close()

                        # # Save label mask
                        # fig, ax = plt.subplots()
                        # ax.imshow(label_combined_mask / label_combined_mask.max())  # Normalize for visualization
                        # ax.axis('off')
                        # plt.savefig(os.path.join(final_vis_path, f'label_mask.png'), bbox_inches='tight', pad_inches=0)
                        # plt.close()

    return


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_overlay_with_boundary(img, mask, save_path, color=(255, 0, 0), alpha=0.3, boundary_color=(255, 255, 0), boundary_thickness=2):
    """
    保存带有边界高亮的叠加图像。
    
    :param img: 原始图像 (H, W, 3)，范围 [0, 1]
    :param mask: mask 图像 (H, W, 3)，范围 [0, 1]
    :param save_path: 保存路径
    :param color: 覆盖的颜色 (RGB)
    :param alpha: 叠加透明度
    :param boundary_color: 边界高亮颜色 (RGB)
    :param boundary_thickness: 边界线条厚度
    """
    overlay = (1 - alpha) * img + mask * alpha
    overlay = (overlay * 255).astype(np.uint8)

    # 将 RGB 转换为 BGR，便于使用 OpenCV
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    mask_gray = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # 查找轮廓
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_bgr, contours, -1, boundary_color[::-1], boundary_thickness)  # RGB -> BGR

    # 保存高亮边界后的叠加图
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    plt.imsave(save_path, overlay_rgb)
    print(f"✅ 保存叠加图及边界高亮: {save_path}")





def apply_color_mask(mask, color):
    # 创建彩色掩码 (H, W, 3)，每个像素值应用指定颜色
    color_mask = np.zeros((*mask.shape, 3))
    for i in range(3):  # RGB 通道
        color_mask[..., i] = mask * color[i]  # 仅对非零像素应用颜色
    return color_mask
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
    
# def train_sam_depth(args, net: nn.Module, optimizer1, optimizer2, train_loader, epoch):
#     epoch_loss = 0
#     epoch_prompt_loss = 0
#     epoch_non_prompt_loss = 0
    
  
#     net.train()
#     if optimizer1 is not None:
#         optimizer1.zero_grad()
#     if optimizer2 is not None:
#         optimizer2.zero_grad()
    
#     video_length = args.video_length
#     GPUdevice = torch.device('cuda:' + str(args.gpu_device))
#     prompt = args.prompt
#     prompt_freq = args.prompt_freq

   
#     lossfunc = DepthLoss(loss='l1')
#     with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
#         for pack in train_loader:
#             torch.cuda.empty_cache()
#             imgs_tensor = pack['image']
#             depth_dict = pack['depth']  

#             if prompt == 'click':
#                 pt_dict = pack['pt']
#                 point_labels_dict = pack['p_label']
#             elif prompt == 'bbox':
#                 bbox_dict = pack['bbox']

#             imgs_tensor = imgs_tensor.squeeze(0).to(dtype=torch.float32, device=GPUdevice)
#             train_state = net.train_init_state(imgs_tensor=imgs_tensor)

#             prompt_frame_id = list(range(0, video_length, prompt_freq))
#             obj_list = list(depth_dict[prompt_frame_id[0]].keys())

#             if len(obj_list) == 0:
#                 continue

#             with torch.cuda.amp.autocast():
#                 for id in prompt_frame_id:
#                     for ann_obj_id in obj_list:
#                         try:
#                             if prompt == 'click':
#                                 points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
#                                 labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
#                                 net.train_add_new_points(
#                                     inference_state=train_state,
#                                     frame_idx=id,
#                                     obj_id=ann_obj_id,
#                                     points=points,
#                                     labels=labels,
#                                     clear_old_points=False,
#                                 )
#                             elif prompt == 'bbox':
#                                 bbox = bbox_dict[id][ann_obj_id]
#                                 net.train_add_new_bbox(
#                                     inference_state=train_state,
#                                     frame_idx=id,
#                                     obj_id=ann_obj_id,
#                                     bbox=bbox.to(device=GPUdevice),
#                                     clear_old_points=False,
#                                 )
#                         except KeyError:
#                             net.train_add_new_mask(
#                                 inference_state=train_state,
#                                 frame_idx=id,
#                                 obj_id=ann_obj_id,
#                                 mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
#                             )
                
#                 video_segments = {}
#                 for out_frame_idx, out_obj_ids, out_depth_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
#                     video_segments[out_frame_idx] = {
#                         out_obj_id: out_depth_logits[i] for i, out_obj_id in enumerate(out_obj_ids)
#                     }

#                 loss = 0
#                 for id in range(video_length):
#                     for ann_obj_id in obj_list:
#                         pred_depth = video_segments[id][ann_obj_id].unsqueeze(0)
#                         try:
#                             true_depth = depth_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
#                         except KeyError:
#                             true_depth = torch.zeros_like(pred_depth).to(device=GPUdevice)

#                         obj_loss = lossfunc(pred_depth, true_depth)
#                         loss += obj_loss.item()

#                 loss = loss / video_length / len(obj_list)
#                 pbar.set_postfix(**{'loss (batch)': loss})
#                 epoch_loss += loss

#                 loss.backward()
#                 optimizer1.step()
#                 if optimizer2 is not None:
#                     optimizer2.step()

#                 optimizer1.zero_grad()
#                 if optimizer2 is not None:
#                     optimizer2.zero_grad()

#                 net.reset_state(train_state)
#             pbar.update()

#     return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)

# def val_sam_depth(args, val_loader, epoch, net: nn.Module):
    
#     net.eval()

#     n_val = len(val_loader)
#     tot_loss = 0
#     prompt_freq = args.prompt_freq

#     lossfunc = torch.nn.L1Loss()
#     prompt = args.prompt

#     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#         for pack in val_loader:
#             imgs_tensor = pack['image']
#             depth_dict = pack['depth']  # 深度图信息

#             if prompt == 'click':
#                 pt_dict = pack['pt']
#                 point_labels_dict = pack['p_label']
#             elif prompt == 'bbox':
#                 bbox_dict = pack['bbox']

#             if len(imgs_tensor.size()) == 5:
#                 imgs_tensor = imgs_tensor.squeeze(0)

#             frame_id = list(range(imgs_tensor.size(0)))
#             train_state = net.val_init_state(imgs_tensor=imgs_tensor)
#             prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
#             obj_list = list(depth_dict[frame_id[0]].keys())

#             if len(obj_list) == 0:
#                 continue

#             with torch.no_grad():
#                 for id in prompt_frame_id:
#                     for ann_obj_id in obj_list:
#                         try:
#                             if prompt == 'click':
#                                 points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
#                                 labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
#                                 net.train_add_new_points(
#                                     inference_state=train_state,
#                                     frame_idx=id,
#                                     obj_id=ann_obj_id,
#                                     points=points,
#                                     labels=labels,
#                                     clear_old_points=False,
#                                 )
#                             elif prompt == 'bbox':
#                                 bbox = bbox_dict[id][ann_obj_id]
#                                 net.train_add_new_bbox(
#                                     inference_state=train_state,
#                                     frame_idx=id,
#                                     obj_id=ann_obj_id,
#                                     bbox=bbox.to(device=GPUdevice),
#                                     clear_old_points=False,
#                                 )
#                         except KeyError:
#                             net.train_add_new_mask(
#                                 inference_state=train_state,
#                                 frame_idx=id,
#                                 obj_id=ann_obj_id,
#                                 mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
#                             )

#                 video_segments = {}
#                 for out_frame_idx, out_obj_ids, out_depth_logits in net.propagate_in_video(train_state, start_frame_idx=0):
#                     video_segments[out_frame_idx] = {
#                         out_obj_id: out_depth_logits[i] for i, out_obj_id in enumerate(out_obj_ids)
#                     }

#                 val_loss = 0
#                 for id in frame_id:
#                     for ann_obj_id in obj_list:
#                         pred_depth = video_segments[id][ann_obj_id].unsqueeze(0)
#                         try:
#                             true_depth = depth_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
#                         except KeyError:
#                             true_depth = torch.zeros_like(pred_depth).to(device=GPUdevice)

#                         val_loss += lossfunc(pred_depth, true_depth).item()

#                 total_num = len(frame_id) * len(obj_list)
#                 val_loss = val_loss / total_num
#                 tot_loss += val_loss

#             net.reset_state(train_state)
#             pbar.update()

#     return tot_loss / n_val, tuple([0]*2)
