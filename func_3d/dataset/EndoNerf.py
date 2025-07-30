import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from func_3d.utils import random_click, generate_bbox

class EndoNeRF(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):
        self.clip_list = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        if mode == 'train':
            self.clip_list = self.clip_list[:1]
        else:
            self.clip_list = self.clip_list[1:]
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation

        if mode == 'train':
            self.video_length = args.video_length
        else:
            self.video_length = None
        self.img_height = 512
        self.img_width = 640

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, index):
        point_label = 1

        clip_folder = self.clip_list[index]
        img_folder = os.path.join(self.data_path, clip_folder, 'images')
        mask_folder = os.path.join(self.data_path, clip_folder, 'gt_masks')
        depth_folder = os.path.join(self.data_path, clip_folder, 'depth')  # Depth folder

        img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
        mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
        depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.depth.png')])  # Depth file naming pattern

        num_frames = len(img_files)

        data_seg_3d = np.zeros((self.img_height, self.img_width, num_frames))
        for i, mask_file in enumerate(mask_files):
            mask_frame = np.array(Image.open(os.path.join(mask_folder, mask_file)).convert('L'))
            data_seg_3d[..., i] = mask_frame

        # Remove empty frames
        for i in range(num_frames):
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j + 1]
                break
        num_frames = data_seg_3d.shape[-1]

        if self.video_length is None:
            video_length = int(num_frames / 4)
        else:
            video_length = self.video_length

        if num_frames > video_length and self.mode == 'train':
            starting_frame = np.random.randint(0, num_frames - video_length + 1)
        else:
            starting_frame = 0

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        depth_tensor = torch.zeros(video_length, 1, self.img_size, self.img_size)  # Depth tensor
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img_file = img_files[frame_index + starting_frame_nonzero]
            depth_file = depth_files[frame_index + starting_frame_nonzero]  # Match depth file with image
            img = Image.open(os.path.join(img_folder, img_file)).convert('RGB')
            depth_img = Image.open(os.path.join(depth_folder, depth_file)).convert('L')  # Load depth as grayscale
            mask = data_seg_3d[..., frame_index]

    
          
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

            depth_img = depth_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            depth_img = torch.from_numpy(np.array(depth_img)).unsqueeze(0).float() 

            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}

            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')

            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = torch.tensor(obj_mask).int()
                obj_mask = torch.nn.functional.interpolate(obj_mask.unsqueeze(0).unsqueeze(0).float(), size=(self.img_size, self.img_size), mode='nearest').int()
                obj_mask = obj_mask.squeeze(0).int()

                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            depth_tensor[frame_index - starting_frame, :, :, :] = depth_img  # Assign depth
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict

            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': clip_folder}

        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'depths': depth_tensor,  # Add depths to return
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'depths': depth_tensor,  # Add depths to return
                'image_meta_dict': image_meta_dict,
            }
