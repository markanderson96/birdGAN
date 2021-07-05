import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import pytorch_lightning as pl
import numpy as np
from typing import Any

class Generator(nn.Module):
    """
    Generator module takes noise, generates output

    Args:
    - conf: configuration dictionary

    Returns:
    - img: output image
    """

    def __init__(self, 
                latent_dim: int, 
                feature_maps: int, 
                image_channels: int
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.img_channels = image_channels

        def _gen_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            bias: bool = False,
            last_block: bool = False,
        ) -> nn.Sequential:
            if not last_block:
                gen_block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            else:
                gen_block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                    nn.Tanh(),
                )

            return gen_block
        
        scale = self.feature_maps // 8
        layers = [_gen_block(self.latent_dim, self.feature_maps * scale, kernel_size=4, stride=1, padding=0)]
        while (scale // 2) >= 1:
            layers.append(_gen_block(self.feature_maps * scale, self.feature_maps * (scale // 2)))
            scale = scale // 2        
        layers.append(_gen_block(self.feature_maps, self.img_channels, last_block=True))

        self.model = nn.Sequential(*layers)
        print(self.model)

    def forward(self, noise: Tensor) -> Tensor:
        return self.model(noise)

class Discriminator(nn.Module):
    """
    Discriminator module to decide whether image is real or generated

    Args:
    - conf: configuration dictionary

    Returns:
    - validity: how valid the image is
    """

    def __init__(self, 
                 feature_maps: int, 
                 image_channels: int
    ) -> None:
        super().__init__()
        
        self.feature_maps = feature_maps
        self.img_channels = image_channels
        

        def _discriminator_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            bias: bool = False,
            batch_norm: bool = True,
            last_block: bool = False,
        ) -> nn.Sequential:
            if not last_block:
                disc_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                    nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                disc_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                    nn.Sigmoid(),
                )

            return disc_block

        scale = 1
        layers = [_discriminator_block(self.img_channels, self.feature_maps * scale, batch_norm=False)]
        while (scale * 2) <= (self.feature_maps // 8):
            layers.append(_discriminator_block(self.feature_maps * scale, self.feature_maps * (scale * 2)))
            scale = scale * 2   
        layers.append(_discriminator_block(self.feature_maps * scale, 1, kernel_size=4, stride=1, padding=0, last_block=True))

        self.model = nn.Sequential(*layers)
        print(self.model)

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image).view(-1, 1).squeeze()

class DCGAN(pl.LightningModule):
    def __init__(
        self,
        beta1: float = 0.5,
        feature_maps_g: int = 64,
        feature_maps_d: int = 64,
        image_channels: int = 1,
        latent_dim: int = 100,
        lr: float = 0.0002,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # the two networks
        self.generator = Generator(self.hparams.latent_dim,
                                   self.hparams.feature_maps_g,
                                   self.hparams.image_channels)
        self.discriminator = Discriminator(self.hparams.feature_maps_d,
                                           self.hparams.image_channels)
        self.criterion = nn.BCELoss()

        # apply random weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        # to get output
        self.validation_z = self._get_noise(16, self.hparams.latent_dim)

    def forward(self, z):
        z = z.view(*z.shape, 1, 1)
        return self.generator(z)
    
    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        result = None
        # train the generator
        if optimizer_idx == 0:
            result = self._generator_step(imgs)

        # train discriminator
        if optimizer_idx == 1:
            # measure the ability of discriminator to classify real from generated
            result = self._discriminator_step(imgs)
        
        return result

    def _generator_step(self, imgs):
        fake_gt = torch.ones(imgs.size(0), 1).squeeze()
        fake_gt = fake_gt.type_as(imgs)

        # get the loss
        fake_pred = self._get_fake_pred(imgs)
        
        g_loss = self.criterion(fake_pred, fake_gt)
        self.log("loss/generator", g_loss, on_epoch=True)
        return g_loss
    
    def _discriminator_step(self, imgs):
        # real images
        real_pred = self.discriminator(imgs)
        real_gt = torch.full_like(real_pred, 0.9)
        real_loss = self.criterion(real_pred, real_gt)
        
        # fake images
        fake_pred = self._get_fake_pred(imgs)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)
        
        # take the average of these losses
        d_loss = (real_loss + fake_loss)
        self.log("loss/discriminator", d_loss, on_epoch=True)

        return d_loss

    def _get_fake_pred(self, real: Tensor) -> Tensor:
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred

    def _get_noise(self, n_samples: int, latent_dim: int) -> Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = (self.hparams.beta1, 0.999)

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0][0].weight)
        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=4)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)