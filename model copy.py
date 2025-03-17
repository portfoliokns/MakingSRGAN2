import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual

class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.upsample1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixelshuffle1 = nn.PixelShuffle(2)
        self.upsample2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixelshuffle2 = nn.PixelShuffle(2)
        
        self.conv3 = nn.Conv2d(64, input_channels, kernel_size=9, padding=4)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.prelu(self.conv1(x))
        residual = x
        x = self.res_blocks(x)
        x = self.bn2(self.conv2(x)) + residual
        x = self.pixelshuffle1(self.upsample1(x))
        x = self.pixelshuffle2(self.upsample2(x))
        x = self.tanh(self.conv3(x))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, img_size=96):
        super(Discriminator, self).__init__()

        def conv_block(in_channels, out_channels, stride=1, bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # SRGAN 論文準拠の構成（畳み込み + BatchNorm + LeakyReLU）
        self.model = nn.Sequential(
            conv_block(input_channels, 64, stride=1, bn=False),  # 最初の層はBatchNormなし
            conv_block(64, 64, stride=2),
            conv_block(64, 128, stride=1),
            conv_block(128, 128, stride=2),
            conv_block(128, 256, stride=1),
            conv_block(256, 256, stride=2),
            conv_block(256, 512, stride=1),
            conv_block(512, 512, stride=2)
        )

        # 特徴マップのサイズ計算
        self.fc_input_size = self._get_fc_input_size(img_size)
        self.fc = nn.Linear(self.fc_input_size, 1)

    def _get_fc_input_size(self, img_size):
        """ダミーの入力を畳み込み層に通して最終的な特徴マップのサイズを計算"""
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size, img_size)
            # print(f"先特徴マップサイズ: {x.shape}")
            x = self.model(x)
            # print(f"特徴マップサイズ: {x.shape}")
            return x.numel()

    def forward(self, x):
        x = self.model(x)
        # print(f"Flatten前の特徴マップのサイズ: {x.shape}") 
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"Flatten後のサイズ: {x.shape}")
        x = self.fc(x)
        return x