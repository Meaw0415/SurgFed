import os
import numpy as np
import torch
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox

class CholecSeg8K(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):
        """
        Initialize the CholecSeg8K dataset with prompt options.
        
        Args:
            data_path (str): Root directory of the dataset.
            mode (str): 'train' or 'test', determines which video folders to use.
            transform: Optional image transformations.
            prompt (str): Type of prompt used ('click' or 'bbox').
            img_size (int): Size to which the images will be resized.
            video_limit (int): The number of videos to use for training (first N videos are for training).
            seed (int): Random seed for reproducibility.
            variation (int): Variation in bounding box generation.
        """
        self.data_path = data_path
        self.mode = mode
        self.img_size = args.image_size
        self.video_length = args.video_length if mode == 'train' else None  # 引入 video_length
        self.prompt = prompt
        self.transform = transform
        self.seed = seed
        self.variation = variation

        all_videos = sorted([d for d in os.listdir(self.data_path) if d.startswith('video')])
        if mode == 'train':
            self.video_list = all_videos[:13]  # First 13 videos for training
        else:
            self.video_list = all_videos[13:]  # Last 4 videos for testing

        # Collect clip paths from each video
        self.clip_paths = []
        for video in self.video_list:
            video_path = os.path.join(self.data_path, video)
            for clip in sorted(os.listdir(video_path)):
                clip_path = os.path.join(video_path, clip)
                self.clip_paths.append(clip_path)

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, index):
        """
        Retrieve a clip (sequence of frames) and its corresponding masks.

        Args:
            index: The index of the clip to retrieve.

        Returns:
            A dictionary containing image tensors, masks, prompt information, and metadata.
        """
        point_label = 1  # Click label default value
        resized_size = (self.img_size, self.img_size)
        clip_path = self.clip_paths[index]
        img_files = sorted([f for f in os.listdir(clip_path) if f.endswith('.npy') and not f.endswith('_mask.npy')])

        num_frames = len(img_files)

        
        if self.video_length is None:
            video_length = int(num_frames / 4)
        else:
            video_length = self.video_length
        if num_frames > video_length and self.mode == 'train':
            starting_frame = np.random.randint(0, num_frames - video_length + 1)
        else:
            starting_frame = 0

        # Initialize tensors to hold the images and masks
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img_file = img_files[frame_index]
            img_path = os.path.join(clip_path, img_file)
            mask_path = img_path.replace('.npy', '_mask.npy')

            # Load the image and mask from .npy files
            img = np.load(img_path)
            mask = np.load(mask_path)

            # Convert image to tensor and resize it
            img = torch.tensor(img).permute(2, 0, 1)  #  (C, H, W)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=resized_size, mode='bilinear', align_corners=False).squeeze(0)
            
            # Convert mask to tensor and extract unique objects
            mask = torch.tensor(mask)
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}

            # Handle bbox and click prompts
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Unrecognized prompt type')

            # Process each object in the mask
            for obj in obj_list:
                obj_mask = (mask == obj).int()  # Generate binary mask for the object
                obj_mask = torch.nn.functional.interpolate(obj_mask.unsqueeze(0).unsqueeze(0).float(), size=resized_size, mode='nearest').int().squeeze(0)

                # Store the resized mask
                diff_obj_mask_dict[obj] = obj_mask

                # Handle click or bbox prompts
                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

            # Store the image and mask tensors
            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict

            # Store bbox or click prompt data
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        # Return meta information along with the images and masks
        image_meta_dict = {'filename_or_obj': clip_path}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }
