import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision.uitls import save_image

from generator import Generator
from discriminator import Discriminator
from dataset import PhotoToVanggogfDataset
from utils import Utils
from config import Config
from tqdm import tqdm

config = Config()
utils = Utils()


class Trainer:
    def train_epoch(self, 
                    config, 
                    disc_V, 
                    gen_V, 
                    disc_R, 
                    gen_R, 
                    trainloader, 
                    valloader, 
                    opt_disc, 
                    opt_gen, 
                    l1_loss, 
                    bce_loss, 
                    d_scalar = torch.cuda.amp.GradScaler(),
                    g_scaler = torch.cuda.amp.GradScaler()
                    ):
        
        discriminator_loss_epoch = 0
        generator_loss_epoch = 0

        loader = tqdm(trainloader, colour="blue")
        for idx, (real_img, vanggogf_img) in enumerate(loader):
            real_img = real_img.to(config.DEVICE)
            vanggogf_img = vanggogf_img.to(config.DEVICE)

            # training discriminator
            with torch.cuda.amp.autocast:

                # disc_V loss --> true vanggogf
                fake_v = gen_V(real_img)
                disc_v_fake_score = disc_V(fake_v.detach())
                disc_v_real_score = disc_V(vanggogf_img)
                disc_v_fake_loss = bce_loss(disc_v_fake_score, torch.zeros_like(disc_v_fake_score))
                disc_v_real_loss = bce_loss(disc_v_real_score, torch.ones_like(disc_v_real_score))
                disc_v_loss = disc_v_real_loss + disc_v_fake_loss

                # disc_R loss --> true real
                fake_r = gen_R(vanggogf_img)
                disc_r_fake_score = disc_R(fake_r.detach())
                disc_r_real_score = disc_R(real_img)
                disc_r_fake_loss = bce_loss(disc_r_fake_score, torch.zeros_like(disc_r_fake_score))
                disc_r_real_loss = bce_loss(disc_r_real_score, torch.ones_like(disc_r_real_score))
                disc_r_loss = disc_r_real_loss + disc_r_fake_loss
                
                disc_loss = (disc_v_loss + disc_r_loss) / 2
                discriminator_loss_epoch += disc_loss.item()
            
            opt_disc.zero_grad()
            d_scaler.scale(disc_loss).backward(retain_graph=True)
            d_scaler.step(opt_disc)
            d_scaler.update()

            # training generator
            while torch.cuda.amp.autocast():
                disc_fake_v = disc_V(fake_v)
                disc_fake_r = disc_R(fake_r)

                # normal gan loss
                gen_loss_v = bce_loss(disc_fake_v, torch.ones_like(disc_fake_v))
                gen_loss_r = bce_loss(disc_fake_r, torch.ones_like(disc_fake_r))

                # cycle loss
                cycle_loss_v = l1_loss(real_img, gen_R(fake_v))
                cycle_loss_r = l1_loss(vanggogf_img, gen_V(fake_r))

                # identity loss
                indentity_loss_v = l1_loss(vanggogf_img, gen_V(vanggogf_img))
                indentity_loss_r = l1_loss(real_img, gen_R(real_img))

                gen_loss = gen_loss_v + gen_loss_r + 
                           (cycle_loss_v + cycle_loss_r) * config.LAMBDA_CYCLE + 
                           (identity_loss_v + indentity_loss_r) * config.LAMBDA_INDENTITY
                generator_loss_epoch = gen_loss.item()

            opt_gen.zero_grad()
            g_scalar.scale(gen_loss).backward()
            g_scalar.step(opt_gen)
            g_scalar.update()

            if not idx%200:
                save_image(fake_anime*0.5+0.5,f"{config.VANGGOGF_SAVED_IMAGES}/{idx}.png")
                save_image(fake_photo*0.5+0.5,f"{config.PHOTO_SAVED_IMAGES}/{idx}.png")

            loader.set_postfix(
                disc_loss = f"{disc_loss.item():.4f}",
                gen_loss = f"{gen_loss.item():.4f}"
            )

        return discriminator_loss_epoch/len(trainloader),generator_loss_epoch/len(trainloader)

    def train():
        disc_V = Discriminator(in_channels=3).to(config.DEVICE)
        gen_V = Generator(in_channels=3).to(config.DEVICE)

        disc_R = Discriminator(in_channels=3).to(config.DEVICE)
        gen_R = Generator(in_channels=3).to(config.DEVICE)
        
        opt_disc = torch.optim.Adam(
            list(disc_V.parameters()) + list(disc_R.parameters()),
            lr = config.LEARNING_RATE,
            betas = (config.BETA_1, config.BETA_2)
        )
        opt_gen = torch.optim.Adam(
            list(gen_V.parameters()) + list(gen_R.parameters()),
            lr = config.LEARNING_RATE,
            betas = (config.BETA_1, config.BETA_2)
        )

        l1_loss = nn.L1Loss()
        bce_loss = nn.BCEWithLogitsLoss()
        if config.LOAD_MODEL:
            utils.load(
                disc_V,
                config.CHECKPOINT_DIS_VANGGOGH
            )

            utils.load(
                disc_R,
                config.CHECKPOINT_DIS_PHOTO
            )

            utils.load(
                gen_V,
                CHECKPOINT_GEN_VANGGOGH
            )

            utils.load(
                gen_R,
                CHECKPOINT_GEN_PHOTO
            )
            
        
        try:
            trainset = PhotoToVanggogfDataset(transform=True)
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size = config.BATCH_SIZE,
                shuffle = True
            )

            valset = PhotoToVanggogfDataset(transform=False)
            valloader = torch.utils.DataLoader(
                valset,
                batch_size = config.BATCH_SIZE,
                shuffle = False
            )
        except ValueError as e:
            print(e)
            return
        
        generator_loss = list()
        discriminator_loss = list()
        print("__Training started__")

        for epoch in range(config.NUM_EPOCHS):
            print(f"Epoch{epoch + 1}/{config.NUM_EPOCHS}")
            
            gen_loss, disc_loss = self.train_epoch(
                config = config,
                disc_vanggogh = disc_V, 
                gen_vanggogh = gen_V, 
                disc_real = disc_R,  
                gen_real = gen_R, 
                trainloader = trainloader, 
                valloader = valloader, 
                opt_disc = opt_disc, 
                opt_gen = opt_gen, 
                l1_loss = ll_loss, 
                bce_loss = bce_loss, 
            )

            generator_loss.append(gen_loss)
            discriminator_loss.append(disc_loss)

            if config.SAVE_MODEL:
                if not os.path.isdir(f'VangGoghGAN/checkpoints'):
                    os.mkdir(f'VangGoghGAN/checkpoints')
                
            utils.save(
                disc_V,
                epoch,
                config.CHECKPOINT_DIS_VANGGOGH
            )
            utils.save(
                disc_R,
                epoch,
                config.CHECKPOINT_DIS_PHOTO
            )
            utils.save(
                gen_V,
                epoch,
                CHECKPOINT_GEN_VANGGOGH
            )
            utils.save(
                gen_R,
                epoch,
                CHECKPOINT_GEN_PHOTO
            )

        print("__Training Complete__")
       
        print("__plotting loss curves__")
        plt.figure(figsize=(30,30))
        plt.plot(generator_loss,color="red")
        plt.plot(discriminator_loss,color='blue')
        plt.legend(['gen_loss','disc_loss'])
        plt.title('LOSS vs EPOCH',fontdict={'fontsize':10})
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.xticks(range(0, config.NUM_EPOCHS+1 , 1),fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()
        
        
