from torchvision import transforms
import os
from typing import Callable
import torch


class Config:

    LEARNING_RATE:float = 0.0002
    BETA_1:float = 0.5
    BETA_2:float = 0.999
    LAMBDA_CYCLE:int = 10
    LAMBDA_IDENTITY:int = 5
    NUM_EPOCHS:int = 20
    BATCH_SIZE:int = 1

    SAVE_MODEL:bool = True
    LOAD_MODEL:bool = False

    CHECKPOINT_GEN_VANGGOGH:Callable[..., str] = lambda self,creator:f'VangGoghGAN/checkpoints/gen_VangGogh.pth'
    CHECKPOINT_DIS_VANGGOGH:Callable[..., str] = lambda self,creator:f'VangGoghGAN/checkpoints/dis_VangGogh.pth'
    CHECKPOINT_GEN_PHOTO:Callable[..., str] = lambda self,creator:f'VangGoghGAN/checkpoints/gen_photo.pth'
    CHECKPOINT_DIS_PHOTO:Callable[..., str] = lambda self,creator:f'VangGoghGANcheckpoints/dis_photo.pth'

    DEVICE:str = "cuda" if torch.cuda.is_available() else "cpu"
    VANGGOGF_SAVED_IMAGES:str = "VangGoghGAN/gen_VangGogh"
    PHOTO_SAVED_IMAGES:str = "VangGoghGAN/gen_photo"

    preprocess:transforms.transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])