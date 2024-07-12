# Vang Gogh Style Transfer with Cycle GAN

This project uses a Cycle GAN model with identity loss to translate natural images into Vang Gogf-style images.

<table align="left">
  <td>
    <a href="https://www.kaggle.com/code/icode100/cyclegan" target="_parent"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Colab"/></a>
  </td>
</table>

<br>

## Table of Contents

- [Sample Generations](#sample)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)


## Sample
These are the outputs samples from the model along with the inputs.
![WhatsApp Image 2024-07-12 at 23 21 55_663b2f9b](https://github.com/user-attachments/assets/6855bb74-2e6a-4e8b-b461-3154fdb15b6b)


## Installation

1. Clone the repository:
   bash
   
  ` https://github.com/mahathibodela/Photo_to_VangGogh_style_converterr.git`
   

3. Install the required dependencies:
   `pip install -r requirements.txt`
   

## Usage

### Pretrained Model

Download the pretrained model from this link for [Vang Gogf's](https://github.com/icode100/AniCyGAN/blob/main/AnimeGAN/checkpoints/Hayao/gen_animation.pth) style 

### Generate Vang Gogf Style Images

1. Visit my [Application](https://phototovanggoghstyleconverterr-ehtlpizn6ajamk3vcdjtos.streamlit.app/).
2. On the Home page upload the image
3. Click on generate image button

## Dataset

Datasets used are [Vang Gogf images](https://www.kaggle.com/datasets/icode100/cycleganvangogf/data), [Real images](https://www.kaggle.com/competitions/gan-getting-started/data?select=photo_jpg)
## Training

To look for the training references look up for the `code` directory and the jupyter notebooks in `training_notebooks` directory.
