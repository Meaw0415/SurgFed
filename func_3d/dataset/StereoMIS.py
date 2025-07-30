import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from func_3d.utils import random_click, generate_bbox

class StereoMIS(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):
        """
        Initialize the StereoMIS dataset with prompt options and clip-level sampling.
        
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
        # Gather clip directories
        clip_list = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

        if mode == 'train':
            self.clip_list = [clip for clip in clip_list if 'P1' in clip]  # 'P1' folders are for training
            self.scale = 250
        else:
            self.clip_list = [clip for clip in clip_list if 'P1' not in clip]  # Other folders are for testing
            self.scale = 250

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

    def __len__(self):
        """
        Return the number of clips.
        """
        return len(self.clip_list)

    def __getitem__(self, index):
        """
        Sample frames from a clip and process them.
        
        Args:
            index: The index of the clip.
        
        Returns:
            A dictionary containing image tensor, depth tensor, masks, prompt information, and metadata.
        """
        point_label = 1

        # Get clip folder and corresponding image, mask, and depth directories
        clip_folder = self.clip_list[index]
        img_folder = os.path.join(self.data_path, clip_folder, '1_video_frames')
        mask_folder = os.path.join(self.data_path, clip_folder, 'masks')
        depth_folder = os.path.join(self.data_path, clip_folder, 'depths')

        # Get all image files within the clip
        img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('l.png')])

        # Initialize tensors for images, depths, and masks
        img_tensor = torch.zeros(self.video_length, 3, self.img_size, self.img_size)
        depth_tensor = torch.zeros(self.video_length, 1, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        # Determine starting frame for sampling
        num_frames = len(img_files)
        if num_frames > self.video_length and self.mode == 'train':
            starting_frame = np.random.randint(0, num_frames - self.video_length + 1)
        else:
            starting_frame = 0

        for frame_index in range(starting_frame, starting_frame + self.video_length):
            img_file = img_files[frame_index]

            # Generate corresponding mask and depth file names based on the image file name
            mask_file = img_file  # Assuming the mask file has the same name as the image file
            depth_file = img_file + '.npy'  # Assuming the depth file is a .npy file with the same base name

            # Load image, mask, and depth
            img = Image.open(os.path.join(img_folder, img_file)).convert('RGB')
            mask = Image.open(os.path.join(mask_folder, mask_file)).convert('L')
            depth = np.load(os.path.join(depth_folder, depth_file))

            # Resize and convert to tensor
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).int()

            depth = torch.from_numpy(depth).unsqueeze(0).float()
            depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear').squeeze(0)
           
            # Store in tensors/dictionaries
            img_tensor[frame_index - starting_frame, :, :, :] = img
            depth_tensor[frame_index - starting_frame, :, :, :] = depth*self.scale
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
                obj_mask_temp = (mask == obj)
                obj_mask = torch.tensor(obj_mask_temp).int()
                obj_mask = torch.nn.functional.interpolate(obj_mask.unsqueeze(0).unsqueeze(0).float(), size=(self.img_size, self.img_size), mode='nearest').int()
                obj_mask = obj_mask.squeeze(0).int()

                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

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
