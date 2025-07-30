import os
import numpy as np
import torch
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox

class AutoLaparo(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):
        """
        Initialize the AutoLaparo dataset with prompt options.
        
        Args:
            args: Configuration arguments, including image size and video length.
            data_path: Root directory of the dataset.
            transform: Optional image transformations.
            transform_msk: Optional mask transformations.
            mode: Indicates 'train' or 'test/validation'.
            prompt: Type of prompt used ('click' or 'bbox').
            seed: Random seed for reproducibility.
            variation: Variation in bounding box generation.
        """
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.video_length = args.video_length if mode == 'train' else None
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation

        # Split clips based on the first three characters of the filenames (clip ID)
        if mode == 'train':
            self.clip_list = sorted(set([f[:3] for f in os.listdir(os.path.join(data_path, 'img')) if int(f[:3]) <= 170]))
        else:
            self.clip_list = sorted(set([f[:3] for f in os.listdir(os.path.join(data_path, 'img')) if int(f[:3]) > 170]))
    
    def __len__(self):
        """
        Return the number of clips in the dataset.
        """
        return len(self.clip_list)

    def __getitem__(self, index):
        """
        Retrieve a clip (sequence of frames) and its corresponding masks.

        Args:
            index: The index of the clip to retrieve.

        Returns:
            A dictionary containing image tensors, masks, prompt information (click or bbox), and metadata.
        """
        point_label = 1
        resized_size = (self.img_size, self.img_size)

        # Get the clip prefix, e.g., '001' for clip 001
        clip_prefix = self.clip_list[index]
        img_path = os.path.join(self.data_path, 'img')
        mask_path = os.path.join(self.data_path, 'mask')

        # Get all frame files corresponding to the clip
        frame_files = sorted([f for f in os.listdir(img_path) if f.startswith(clip_prefix)])
        num_frames = len(frame_files)

        # Clip the video length if necessary
        if self.video_length is not None and num_frames > self.video_length:
            starting_frame = np.random.randint(0, num_frames - self.video_length + 1)
        else:
            starting_frame = 0
            self.video_length = num_frames  # If num_frames is less than or equal to video_length

        # Initialize tensors to hold the images and masks
        img_tensor = torch.zeros(self.video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        # Process each frame in the clip
        for frame_index in range(starting_frame, starting_frame + self.video_length):
            frame_file = frame_files[frame_index]
            frame_idx = int(frame_file.split('.')[0][-3:])  # Extract frame index

            # Load the image and mask from .npy files
            img = np.load(os.path.join(img_path, f'{clip_prefix}{frame_idx:03d}.npy'))
            class_mask = np.load(os.path.join(mask_path, f'{clip_prefix}{frame_idx:03d}_mask.npy'))

            # Convert image to tensor and resize it
            img = torch.tensor(img).permute(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=resized_size, mode='bilinear', align_corners=False).squeeze(0)

            # Convert class ID mask to tensor
            class_mask = torch.tensor(class_mask).long()  # (H, W)

            # Extract unique objects from the mask
            obj_list = np.unique(class_mask[class_mask > 0])
            diff_obj_mask_dict = {}

            # Handle different prompt types
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')

            # Process each object in the mask
            for obj in obj_list:
                obj_mask = (class_mask == obj)  # Generate binary mask for the object
                obj_mask = obj_mask.int()

                # Resize the mask to the target size
                obj_mask = torch.nn.functional.interpolate(obj_mask.unsqueeze(0).unsqueeze(0).float(), size=resized_size, mode='nearest').int()
                obj_mask = obj_mask.squeeze(0).int()

                # Store the resized mask
                diff_obj_mask_dict[obj] = obj_mask

                # Handle click or bbox prompts
                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

            # Store image and mask tensors
            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict

            # Store bbox or click prompt data
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        # Return meta information along with the images and masks
        image_meta_dict = {'filename_or_obj': clip_prefix}
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
