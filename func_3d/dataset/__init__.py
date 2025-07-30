from .btcv import BTCV
from .amos import AMOS
from .Endovis2017 import Endovis2017
from .Endovis2018 import Endovis2018
from .AutoLaparo import AutoLaparo
from .cholecseg8k import CholecSeg8K
from .SimCol3D import SimCol3D
from .EndoNerf import EndoNeRF
from .StereoMIS import StereoMIS
from .SCARED import SCARED

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_PATH = {
    # 'btcv': '/data/BTCV',
    # 'amos': '/data/AMOS',
    'Endovis2017': '/mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/Endovis2017/site1',
    'Endovis2018': '/mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/Endovis2018/site2',
    'AutoLaparo': '/mnt/iMVR/zhengf/MTFL/NYU_PAS_Dataset/AutoLaparo',
    # 'cholecseg8k': '/mnt/iMVR/zhengf/FL-Multi/FL-MultiTask_Dataset/cholecseg8k',
    'SCARED': '/mnt/iMVR/zhengf/MTFL/NYU_PAS_Dataset/SCARED',
    'StereoMIS': '/mnt/iMVR/zhengf/MTFL/NYU_PAS_Dataset/StereoMIS',
}

def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'Endovis2017':
        '''Endovis2017 data'''
        endovis2017_train_dataset = Endovis2017(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        endovis2017_test_dataset = Endovis2017(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        nice_train_loader = DataLoader(endovis2017_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(endovis2017_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'Endovis2018':
        '''Endovis2018 data'''
        endovis2018_train_dataset = Endovis2018(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        endovis2018_test_dataset = Endovis2018(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        nice_train_loader = DataLoader(endovis2018_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(endovis2018_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''

    elif args.dataset == 'AutoLaparo':
        '''Autolaparo data'''
        autolaparo_train_dataset = AutoLaparo(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        autolaparo_test_dataset = AutoLaparo(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        nice_train_loader = DataLoader(autolaparo_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(autolaparo_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'cholecseg8k':
        '''cholecseg8k data'''
        cholecseg8k_train_dataset = CholecSeg8K(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        cholecseg8k_test_dataset = CholecSeg8K(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        nice_train_loader = DataLoader(cholecseg8k_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(cholecseg8k_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'SimCol3D':
        '''SimCol3D data'''
        simcol3d_train_dataset = SimCol3D(args, args.data_path, transform = None, mode = 'train')
        simcol3d_test_dataset = SimCol3D(args, args.data_path, transform = None, mode = 'test')

        nice_train_loader = DataLoader(simcol3d_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(simcol3d_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'EndoNerf':
        '''EndoNerf data'''
        endonerf_train_dataset = EndoNeRF(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        endonerf_test_dataset = EndoNeRF(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        nice_train_loader = DataLoader(endonerf_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(endonerf_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'StereoMIS':
        '''StereoMIS data'''
        stereomis_train_dataset = StereoMIS(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        stereomis_test_dataset = StereoMIS(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        nice_train_loader = DataLoader(stereomis_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(stereomis_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'SCARED':
        '''SCARED data'''
        scared_train_dataset = SCARED(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        scared_test_dataset = SCARED(args, DATA_PATH[args.dataset], transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        nice_train_loader = DataLoader(scared_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(scared_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    else:
        raise ValueError('Dataset not supported: {}'.format(args.dataset))
        
    return nice_train_loader, nice_test_loader
