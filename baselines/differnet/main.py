'''This is the repo which contains the original code to the WACV 2022 paper
"Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection"
by Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn and Bastian Wandt.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
from utils import make_dataloaders
import numpy as np
import torch
import os, random
from our_config import get_args
from custom_datasets import *
from torchvision import transforms
from torchvision.transforms.functional import rotate

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def get_random_transforms():
    augmentative_transforms = []
    if c.transf_rotations:
        augmentative_transforms += [transforms.RandomRotation(180)]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                                           saturation=c.transf_saturation)]

    tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                       transforms.Normalize(c.norm_mean, c.norm_std)]

    transform_train = transforms.Compose(tfs)
    return transform_train

def main(args):
    ### set user and dataset path
    if args.user == 'sojin':
        if args.dataset == 'mvtec':
            args.data_path = '/home/sojin/dataset/mvtec'
        elif args.dataset == 'stc':
            args.data_path = './dataset/STC/shanghaitech'
        elif args.dataset == 'multi_mvtec':
            args.data_path = '/home/sojin/dataset/mvtec'
        elif args.dataset == 'video':
            args.data_path = c.video_path
        elif args.dataset == 'toy_example':
            args.data_path = './' #TODO
        else:
            raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    elif args.user == 'juyeon':
        if args.dataset == 'mvtec':
            args.data_path = '/home/juyeon/data/mvtec'
        elif args.dataset == 'stc':
            args.data_path = './data/STC/shanghaitech'
        elif args.dataset == 'multi_mvtec':
            args.data_path = '/home/juyeon/data/mvtec'
        elif args.dataset == 'video':
            args.data_path = args.video_path
        elif args.dataset == 'toy_example':
            args.data_path = './'
        else:
            raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    elif args.user == 'kakao':
        if args.dataset == 'mvtec':
            args.data_path = '/root/dataset/mvtec'
        elif args.dataset == 'stc':
            args.data_path = '/root/dataset/shanghaitech'
        elif args.dataset == 'multi_mvtec':
            args.data_path = '/root/dataset/mvtec'
        elif args.dataset == 'video':
            args.data_path = args.video_path
        elif args.dataset == 'toy_example':
            args.data_path = './'
        else:
            raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    
    args.img_size = c.img_size
    args.crp_size = c.img_size
    args.norm_mean = c.norm_mean
    args.norm_std = c.norm_std
    args.n_transforms = c.n_transforms
    args.n_transforms_test = c.n_transforms_test
    args.transf_rotations = c.transf_rotations
    args.transf_brightness = c.transf_brightness
    args.transf_contrast = c.transf_contrast
    args.transf_saturation = c.transf_saturation
    
    transform_train = get_random_transforms()

    ### load datasets
    if args.dataset == 'mvtec':
        train_dataset = MVTecDataset(args, is_train=True)
        test_dataset  = MVTecDataset(args, is_train=False)
    elif args.dataset == 'stc':
        train_dataset = StcDataset(args, is_train=True)
        test_dataset  = StcDataset(args, is_train=False)
    elif args.dataset == 'multi_mvtec':
        train_dataset = MultiModal_MVTecDataset(transform_train, args, is_train=True)
        test_dataset  = MultiModal_MVTecDataset(transform_train, args, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))

    ### fix seeds
    init_seeds(seed=2022)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_loader, test_loader = make_dataloaders(train_dataset, test_dataset)
    train(args, train_loader, test_loader)

if __name__ == '__main__':
    args = get_args()
    main(args)
