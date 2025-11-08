import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channel, features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channel, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            self._block(features, features*2, 4, 2, 1),
            self._block(features*2, features*4, 4, 2, 1),
            self._block(features*4, features*8, 4, 2, 1),

            nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, z_dim, img_channel, features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(

            self._block(z_dim, features*16, 4, 1, 0),
            self._block(features*16, features*8, 4, 2, 1),
            self._block(features*8, features*4, 4, 2, 1),
            self._block(features*4, features*2, 4, 2, 1),
            
            nn.ConvTranspose2d(
                features * 2, img_channel, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.gen(x)


