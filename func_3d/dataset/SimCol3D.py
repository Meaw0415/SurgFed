import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from func_3d.utils import random_click, generate_bbox

class SimCol3D(Dataset):
    def __init__(self, args, data_path, transform=None, mode='train', seed=None):
        """
        Initialize the SimCol3D dataset for depth prediction.

        Args:
            data_path (str): Root directory of the dataset.
            mode (str): 'train' or 'test', determines which video folders to use.
            transform: Optional image transformations.
            img_size (int): Size to which the images will be resized.
            video_length (int): Number of frames to include in a video sequence.
            seed (int): Random seed for reproducibility.
        """
        self.data_path = data_path
        self.mode = mode
        self.img_size = args.image_size
        self.video_length = args.video_length if mode == 'train' else None  # Video length for train mode
        self.transform = transform
        self.seed = seed

        # Define video clips (three videos, first two for training, last for testing)
        all_videos = sorted(os.listdir(self.data_path))
        if self.mode == 'train':
            self.video_list = all_videos[:2]  # First two videos for training
        else:
            self.video_list = all_videos[2:]  # Last video for testing

        # Collect image and depth file paths from each clip folder
        self.data_pairs = []
        for video in self.video_list:
            video_path = os.path.join(self.data_path, video)
            for clip in sorted(os.listdir(video_path)):
                clip_path = os.path.join(video_path, clip)
                depth_file = os.path.join(clip_path, 'Depth_0000.png')
                img_file = os.path.join(clip_path, 'FrameBuffer_0093.png')
                
                if os.path.exists(depth_file) and os.path.exists(img_file):
                    self.data_pairs.append((img_file, depth_file))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        """
        Retrieve a sequence of images and their corresponding depth maps.

        Args:
            index: The index of the data pair to retrieve.

        Returns:
            A dictionary containing image tensor sequences, depth tensor sequences, and metadata.
        """
        img_path, depth_path = self.data_pairs[index]

        # Load the image and depth map from PNG files
        img = np.array(Image.open(img_path).convert('RGB'))  # Convert image to RGB
        depth = np.array(Image.open(depth_path))  # Depth as single channel
        
        num_frames = len(img) if img.ndim == 4 else 1  # (C, H, W) for single images, (N, C, H, W) for video frames

        # Initialize tensors to hold the images and depth maps
        img_tensor = torch.zeros(min(self.video_length, num_frames), 3, self.img_size, self.img_size)
        depth_tensor = torch.zeros(min(self.video_length, num_frames), 1, self.img_size, self.img_size)

        # Determine the start and end of the frame sequence
        if self.video_length is not None and num_frames > self.video_length:
            starting_frame = np.random.randint(0, num_frames - self.video_length + 1)
        else:
            starting_frame = 0

        video_length = min(self.video_length, num_frames)

        for frame_index in range(starting_frame, starting_frame + video_length):
            if img.ndim == 4:
                img_frame = img[frame_index]
                depth_frame = depth[frame_index]
            else:
                img_frame = img  # Single frame case
                depth_frame = depth

            # Convert image and depth to tensors and resize them
            img_frame = torch.tensor(img_frame).permute(2, 0, 1)  # (C, H, W)
            depth_frame = torch.tensor(depth_frame).unsqueeze(0).float()  # (1, H, W) for depth

            img_frame = torch.nn.functional.interpolate(img_frame.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
            depth_frame = torch.nn.functional.interpolate(depth_frame.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear').squeeze(0)

            # Store in tensors
            img_tensor[frame_index - starting_frame, :, :, :] = img_frame
            depth_tensor[frame_index - starting_frame, :, :, :] = depth_frame

            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
        
            if self.prompt == 'bbox':
                # Entire image as bbox: [(x_min, y_min), (x_max, y_max)]
                diff_obj_bbox_dict[1] = torch.tensor([[0, 0, self.img_size, self.img_size]])  # One bounding box for the whole image
            elif self.prompt == 'click':
                # Click points: choose center point as click
                center_point = torch.tensor([[self.img_size // 2, self.img_size // 2]])  # Center point as click
                point_label = torch.tensor([1])  # Label for click (1 for positive, 0 for negative)

                diff_obj_pt_dict[1] = center_point
                diff_obj_point_label_dict[1] = point_label
            else:
                raise ValueError('Prompt not recognized')

        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return image, depth, and metadata
        return {
            'image': img_tensor,
            'label': depth_tensor,
            'image_meta_dict': {'filename': os.path.basename(img_path)}
        }
