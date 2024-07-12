import torch
import torch.nn as nn
from PIL import Image
from config import Config
from generator import Generator
import matplotlib.pyplot as plt
import numpy as np
import os 


def generateImage(path, model):
    photo_img = Image.open(f'checkPhotos/{path}').convert('RGB')
    photo_processed = config.preprocess(photo_img).unsqueeze(0)
    photo_tensor = model(photo_processed)
    photo = photo_tensor.squeeze(0)

    photo_array = photo.permute(1, 2, 0).detach().cpu().numpy()
    photo_array = (photo_array * 0.5 + 0.5)  

    plt.imshow(photo_array)
    plt.axis('off') 
    plt.show()




config = Config()
generator = Generator(3)
checkpoint = torch.load('VangGoghGAN/checkpoints/gen_VangGogh.pth', map_location=torch.device('cpu'))
generator.load_state_dict(checkpoint["model"])

for file in os.listdir('./checkPhotos'):
    generateImage(file, generator)









# photo_tensor = generateImage('checkPhotos/WhatsApp Image 2024-07-12 at 20.18.23_4e8c6b95.jpg', generator)
# photo = photo_tensor.squeeze(0)

# photo_array = photo.permute(1, 2, 0).detach().cpu().numpy()
# photo_array = (photo_array * 0.5 + 0.5)  

# plt.imshow(photo_array)
# plt.axis('off') 
# plt.show()

# def count_files_in_directory(directory):
#     return len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

# directory = './all'
# print(f'Number of files in directory: {count_files_in_directory(directory)}')

