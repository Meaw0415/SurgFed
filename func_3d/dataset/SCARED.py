import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from func_3d.utils import random_click, generate_bbox

class SCARED(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):
        """
        Initialize the SCARED dataset with prompt options and clip-level sampling.
        
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
        # Gather dataset directories based on mode
        base_folder = 'training' if mode == 'train' else 'testing'
        dataset_list = sorted([d for d in os.listdir(os.path.join(data_path, base_folder)) if d.startswith('dataset')])

        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.video_length = args.video_length
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        self.img_height = 512
        self.img_width = 640
        self.dataset_list = dataset_list
        self.scale = 1 / 256 if mode == 'train' else 1.0 / (256)

        # Collect all keyframes within each dataset
        self.keyframe_paths = []
        for dataset in self.dataset_list:
            dataset_folder = os.path.join(self.data_path, base_folder, dataset)
            keyframe_folders = sorted([kf for kf in os.listdir(dataset_folder) if kf.startswith('keyframe')])
            for keyframe in keyframe_folders:
                self.keyframe_paths.append(os.path.join(dataset_folder, keyframe))

    def __len__(self):
        """
        Return the number of keyframes across all datasets.
        """
        return len(self.keyframe_paths)

    def __getitem__(self, index):
        """
        Sample frames from a keyframe and process them.
        
        Args:
            index: The index of the keyframe.
        
        Returns:
            A dictionary containing image tensor, depth tensor, mask, prompt information, and metadata.
        """
        point_label = 1

        # Get current keyframe folder
        keyframe_folder = self.keyframe_paths[index]
        img_folder = os.path.join(keyframe_folder, 'data', 'left_undistorted')
        depth_folder = os.path.join(keyframe_folder, 'data', 'depthmap_undistorted')

        # Get all image and depth files
        img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
        depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])

        # Determine starting frame for sampling
        num_frames = len(img_files)
        if num_frames > self.video_length and self.mode == 'train':
            # starting_frame = np.random.randint(0, num_frames - self.video_length + 1)
            starting_frame = 0
        else:
            starting_frame = 0

        img_tensor = torch.zeros(self.video_length, 3, self.img_size, self.img_size)
        depth_tensor = torch.zeros(self.video_length, 1, self.img_size, self.img_size)
        mask_dict = {}
        pt_dict = {}
        point_label_dict = {}
        bbox_dict = {}

        # Process each frame in the range
        for frame_index in range(starting_frame, starting_frame + self.video_length):
            img_file = img_files[frame_index]
            depth_file = depth_files[frame_index]

            img = Image.open(os.path.join(img_folder, img_file)).convert('RGB')
            depth = Image.open(os.path.join(depth_folder, depth_file)).convert('L')
            depth = np.array(depth).astype(np.float32) * self.scale

            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

            depth = torch.from_numpy(depth).unsqueeze(0).float()
            depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear').squeeze(0)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            depth_tensor[frame_index - starting_frame, :, :, :] = depth

            # Initialize mask for each frame (assuming a foreground mask in SCARED)
            mask = torch.ones(self.img_size, self.img_size).int()
            obj_list = [1]  # Assuming only one foreground object
            diff_obj_mask_dict = {}

            # Handle prompts for each object in obj_list
            if self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            elif self.prompt == 'bbox':
                diff_obj_bbox_dict = {}

            for obj in obj_list:
                obj_mask = mask.clone()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(obj_mask.numpy(), point_label, seed=self.seed)
                elif self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(obj_mask.numpy(), variation=self.variation, seed=self.seed)

            # Save each frame's mask and prompt data
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict
            elif self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
        image_meta_dict = {'filename_or_obj': keyframe_folder[-20:]}

        # Return dictionary based on the prompt type
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'depths': depth_tensor,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'depths': depth_tensor,
                'image_meta_dict': image_meta_dict,
            }
