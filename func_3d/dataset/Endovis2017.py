import os
import numpy as np
import torch
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox

class Endovis2017(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):

        self.video_list = sorted(set([f.split('frame')[0] for f in os.listdir(os.path.join(data_path, mode, 'image'))]))

        
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
            self.img_height = 1024
            self.img_width = 1280
        else:
            self.video_length = None
            self.img_height = 1080
            self.img_width = 1920

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_height, self.img_width)
        resized_size = (self.img_size, self.img_size)
 
        video_prefix = self.video_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image')
        mask_path = os.path.join(self.data_path, self.mode, 'mask')
        
        
        frame_files = sorted([f for f in os.listdir(img_path) if f.startswith(video_prefix)])
        num_frames = len(frame_files)
        
        
        data_seg_3d = np.zeros((self.img_height, self.img_width, num_frames))
        for i, frame_file in enumerate(frame_files):
            frame_idx = int(frame_file.split('frame')[-1].split('.')[0])  # 提取frame index
            mask_frame = np.load(os.path.join(mask_path, f'{video_prefix}frame{frame_idx:03d}.npy'))
            data_seg_3d[..., i] = mask_frame

        # Remove empty frames
        for i in range(num_frames):
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j+1]
                break
        num_frames = data_seg_3d.shape[-1]

        # Clip Video
        if self.video_length is None:
            video_length = int(num_frames / 4)
        else:
            video_length = self.video_length
        if num_frames > video_length and self.mode == 'train':
            starting_frame = np.random.randint(0, num_frames - video_length + 1)
        else:
            starting_frame = 0
        
    
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            frame_file = frame_files[frame_index + starting_frame_nonzero]
            frame_idx = int(frame_file.split('frame')[-1].split('.')[0])

    
            img = np.load(os.path.join(img_path, f'{video_prefix}frame{frame_idx:03d}.npy'))  
            mask = data_seg_3d[..., frame_index]

  
            img = torch.tensor(img).permute(2, 0, 1)  
            # img = torch.nn.functional.interpolate(img.unsqueeze(0), size=newsize, mode='bilinear', align_corners=False).squeeze(0)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
            

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
                
                # obj_mask = torch.nn.functional.interpolate(obj_mask.unsqueeze(0).float(), size=newsize, mode='nearest').int()
                # print("1",obj_mask.shape) # 1 torch.Size([1024, 1280])
                obj_mask = torch.nn.functional.interpolate(obj_mask.unsqueeze(0).unsqueeze(0).float(), size=(self.img_size, self.img_size), mode='nearest').int()
                obj_mask = obj_mask.squeeze(0).int()
                # print("2",obj_mask.shape) # 2 torch.Size([1, 1024, 1280])
                
                diff_obj_mask_dict[obj] = obj_mask
                
       
                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict    
            
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict
       
        image_meta_dict = {'filename_or_obj': video_prefix}
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


# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from func_3d.utils import random_click, generate_bbox

# class Endovis2017(Dataset):
#     def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', seed=None, variation=0):
#         # 解析文件名，获取所有视频序列
#         self.video_list = sorted(set([f.split('frame')[0] for f in os.listdir(os.path.join(data_path, mode, 'image'))]))

#         # 数据集基本信息
#         self.data_path = data_path
#         self.mode = mode
#         self.prompt = prompt
#         self.img_size = args.image_size
#         self.transform = transform
#         self.transform_msk = transform_msk
#         self.seed = seed
#         self.variation = variation

#         # 视频长度设置
#         if mode == 'train':
#             self.video_length = args.video_length
#             self.img_height = 1024
#             self.img_width = 1280
#         else:
#             self.video_length = None
#             self.img_height = 1080
#             self.img_width = 1920

#         # 在初始化时构建所有帧的索引列表
#         self.frame_index_list = []
#         for video_prefix in self.video_list:
#             img_path = os.path.join(self.data_path, self.mode, 'image')
#             frame_files = sorted([f for f in os.listdir(img_path) if f.startswith(video_prefix)])
#             for frame_file in frame_files:
#                 frame_idx = int(frame_file.split('frame')[-1].split('.')[0])  # 提取 frame index
#                 self.frame_index_list.append((video_prefix, frame_idx))

#     def __len__(self):
#         # 返回总的帧数
#         return len(self.frame_index_list)

#     def __getitem__(self, index):
#         point_label = 1
#         newsize = (self.img_height, self.img_width)
#         resized_size = (self.img_size, self.img_size)

#         # 确保从当前索引开始的片段有足够的帧数
#         video_length = self.video_length if self.video_length is not None else len(self.frame_index_list) // 4
#         if index + video_length > len(self.frame_index_list):
#             index = len(self.frame_index_list) - video_length  # 调整到最后一段可用片段

#         # 初始化张量
#         img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
#         mask_dict = {}
#         point_label_dict = {}
#         pt_dict = {}
#         bbox_dict = {}

#         # 获取视频帧并处理
#         for i in range(video_length):
#             video_prefix, frame_idx = self.frame_index_list[index + i]
            
#             img_path = os.path.join(self.data_path, self.mode, 'image')
#             mask_path = os.path.join(self.data_path, self.mode, 'mask')

#             img = np.load(os.path.join(img_path, f'{video_prefix}frame{frame_idx:03d}.npy'))  # 读取npy格式图像
#             mask = np.load(os.path.join(mask_path, f'{video_prefix}frame{frame_idx:03d}.npy'))  # 读取npy格式掩码

#             img = torch.tensor(img).permute(2, 0, 1)
#             img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)

#             obj_list = np.unique(mask[mask > 0])
#             diff_obj_mask_dict = {}

#             if self.prompt == 'bbox':
#                 diff_obj_bbox_dict = {}
#             elif self.prompt == 'click':
#                 diff_obj_pt_dict = {}
#                 diff_obj_point_label_dict = {}
#             else:
#                 raise ValueError('Prompt not recognized')

#             for obj in obj_list:
#                 obj_mask = mask == obj
#                 obj_mask = torch.tensor(obj_mask).int()
#                 obj_mask = torch.nn.functional.interpolate(obj_mask.unsqueeze(0).unsqueeze(0).float(), size=(self.img_size, self.img_size), mode='nearest').int()
#                 obj_mask = obj_mask.squeeze(0).int()
#                 diff_obj_mask_dict[obj] = obj_mask

#                 if self.prompt == 'click':
#                     diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
#                 if self.prompt == 'bbox':
#                     diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

#             img_tensor[i, :, :, :] = img
#             mask_dict[i] = diff_obj_mask_dict

#             if self.prompt == 'bbox':
#                 bbox_dict[i] = diff_obj_bbox_dict
#             elif self.prompt == 'click':
#                 pt_dict[i] = diff_obj_pt_dict
#                 point_label_dict[i] = diff_obj_point_label_dict

#         image_meta_dict = {'filename_or_obj': video_prefix}

#         # 返回与Endovis2017一致的输出结构
#         if self.prompt == 'bbox':
#             return {
#                 'image': img_tensor,
#                 'label': mask_dict,
#                 'bbox': bbox_dict,
#                 'image_meta_dict': image_meta_dict,
#             }
#         elif self.prompt == 'click':
#             return {
#                 'image': img_tensor,
#                 'label': mask_dict,
#                 'p_label': point_label_dict,
#                 'pt': pt_dict,
#                 'image_meta_dict': image_meta_dict,
#             }
