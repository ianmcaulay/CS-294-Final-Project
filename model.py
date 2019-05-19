import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import get_loader
from torchsummary import summary


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    #    def __init__(self, conv_dim=64, num_speakers=10, repeat_num=6):
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()

        # Source downsampling.
        src_layers = []
        src_layers.append(nn.Conv2d(1, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False))
        src_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        src_layers.append(nn.ReLU(inplace=True))

        curr_dim = conv_dim
        for i in range(2):
            src_layers.append(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
            src_layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            src_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.src_downsample = nn.Sequential(*src_layers)

        # Target downsampling.
        trg_layers = []
        # mb = minibatch size
        # Assume conv_dim = 64 in dimension calculations.
        # (mb, 1, 36, 8192)
        trg_layers.append(nn.Conv2d(1, conv_dim, kernel_size=(3, 9), stride=(1, 4), padding=(1, 4), bias=False))
        trg_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        trg_layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        # (mb, 64, 36, 2048)
        trg_layers.append(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(4, 8), stride=(2, 4), padding=(1, 3), bias=False))
        trg_layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        trg_layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2
        # (mb, 128, 18, 512)
        trg_layers.append(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(4, 8), stride=(2, 4), padding=(1, 3), bias=False))
        trg_layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        trg_layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2
        # (mb, 256, 9, 128)

        trg_layers.append(
            nn.Conv2d(curr_dim, curr_dim, kernel_size=(3, 8), stride=(1, 2), padding=(1, 3), bias=False))
        trg_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
        trg_layers.append(nn.ReLU(inplace=True))
        # (mb, 256, 9, 64)

        trg_layers.append(
            nn.Conv2d(curr_dim, curr_dim, kernel_size=(3, 8), stride=(1, 4), padding=(1, 3), bias=False))
        trg_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
        trg_layers.append(nn.ReLU(inplace=True))
        # (mb, 256, 9, 16)
        trg_layers.append(
            nn.Conv2d(curr_dim, curr_dim, kernel_size=(3, 8), stride=(1, 4), padding=(1, 3), bias=False))
        trg_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
        trg_layers.append(nn.ReLU(inplace=True))
        # (mb, 256, 9, 4)
        trg_layers.append(
            nn.Conv2d(curr_dim, curr_dim, kernel_size=(3, 8), stride=(1, 4), padding=(1, 3), bias=False))
        trg_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
        trg_layers.append(nn.ReLU(inplace=True))
        # (mb, 256, 9, 1)

        self.trg_downsample = nn.Sequential(*trg_layers)

        main_layers = []
        # Multiply dim by 2 since we concat the src and trg vectors, which have the same channel dim by construction.
        curr_dim *= 2

        # Bottleneck layers.
        for i in range(repeat_num):
            main_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            main_layers.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            main_layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            main_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        main_layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        self.main = nn.Sequential(*main_layers)

    def forward(self, src, trg):
        # src_embed has dim (mb, 256, 9, L)
        src_embed = self.src_downsample(src)
        # trg_embed has dim (mb, 256, 9, 1)
        trg_embed = self.trg_downsample(trg)
        trg_embed_tiled = trg_embed.repeat(1, 1, 1, src_embed.size(3))
        # concat has dim (mb, 512, 9, L)
        concat = torch.cat((src_embed, trg_embed_tiled), dim=1)
        return self.main(concat)

    def forward_with_trg_embed(self, src, trg_embed):
        # src_embed has dim (mb, 256, 9, L)
        src_embed = self.src_downsample(src)
        trg_embed_tiled = trg_embed.repeat(1, 1, 1, src_embed.size(3))
        # concat has dim (mb, 512, 9, L)
        concat = torch.cat((src_embed, trg_embed_tiled), dim=1)
        return self.main(concat)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5, num_speakers=10):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num))  # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num))  # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0,
                                  bias=False)  # padding should be 0
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1,
                                       padding=0, bias=False)  # for num_speaker

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_loader('/scratch/sxliu/data_exp/VCTK-Corpus-22.05k/mc/train', 16, 'train', num_workers=1)
    data_iter = iter(train_loader)
    G = Generator().to(device)
    D = Discriminator().to(device)
    for i in range(10):
        mc_real, spk_label_org, acc_label_org, spk_acc_c_org = next(data_iter)
        mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d
        mc_real = mc_real.to(device)  # Input mc.
        spk_label_org = spk_label_org.to(device)  # Original spk labels.
        acc_label_org = acc_label_org.to(device)  # Original acc labels.
        spk_acc_c_org = spk_acc_c_org.to(device)  # Original spk acc conditioning.
        mc_fake = G(mc_real, spk_acc_c_org)
        print(mc_fake.size())
        out_src, out_cls_spks, out_cls_emos = D(mc_fake)
