import torch.nn as nn
import torch.nn.functional as F
import torch


def edge_pad(x, padding):
    # padding = (left, right, top, bottom)
    return F.pad(x, padding, mode='replicate')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()

        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        self.seasonDown = ChannelSupervisionUnit(out_size, out_size)
        self.conv1 = torch.nn.Conv2d(out_size * 2, out_size, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_size, out_size, kernel_size=1)
        self.droupout = nn.Dropout(0.3)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.norm1 = nn.InstanceNorm2d(out_size)

        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, season_feature):
        y1 = self.model(x)
        return y1


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        self.seasonUp = ChannelSupervisionUnitUP(out_size, out_size)
        self.conv1 = torch.nn.Conv2d(out_size * 2, out_size, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_size, out_size, kernel_size=1)
        self.ChannelAttention1 = ChannelAttention(out_size, ratio=16)
        self.norm1 = nn.InstanceNorm2d(out_size)
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.droupout = nn.Dropout(0.3)
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input, season_feature):
        y1 = self.model(x)  # unet上采样
        y7 = torch.cat((y1, skip_input), 1)
        return y7



class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        
        # BatchNorm & Activation
        self.bn = nn.BatchNorm2d(channels)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.001)

        # First Multi-Scale Convolution (3x3 + 5x5)
        self.conv3x3_1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.conv5x5_1 = nn.Conv2d(channels, channels // 2, kernel_size=5, padding=2)
        
        # Second Multi-Scale Convolution (3x3 + 5x5)
        self.conv3x3_2 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.conv5x5_2 = nn.Conv2d(channels, channels // 2, kernel_size=5, padding=2)
        
        # Final 3x3 Convolution
        self.conv3x3_final = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # First Multi-Scale Branch
        y1 = self.leakyrelu(self.bn(self.conv3x3_1(x)))
        y2 = self.leakyrelu(self.bn(self.conv5x5_1(x)))
        y_cat1 = torch.cat([y1, y2], dim=1)  # Concatenate along channels
        y3 = self.leakyrelu(self.bn(self.conv3x3_2(y_cat1)))
        y4 = self.leakyrelu(self.bn(self.conv5x5_2(y_cat1)))
        y_cat2 = torch.cat([y3, y4], dim=1)  # Concatenate along channels
        y_final = self.conv3x3_final(y_cat2)
        y_out = self.leakyrelu(y_final + x)  # Residual connection
        y_out = self.dropout(y_out)

        return y_out


class UPResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(UPResidualBlock, self).__init__()
        self.channels = channels
        self.bn1 = nn.BatchNorm2d(channels)  # 批归一化
        self.bn2 = nn.BatchNorm2d(channels)
        self.droupout = nn.Dropout(0.005)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=7, padding=3)

    def forward(self, x):
        y1 = F.relu(self.bn1(self.conv1(x)))
        y2 = self.bn1(self.conv2(y1))
        y3 = F.relu(y2 + x)
        y4 = self.droupout(y3)
        return y4





class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super(GeneratorUNet, self).__init__()
        drouprate = 0.1
        drouprate2 = 0.1
        self.channel_attention = ChannelAttention_input(in_channels)
        self.down1 = UNetDown(in_channels, 64, normalize=True)
        self.ResidualBlock1 = ResidualBlock(64)
        self.ResidualBlock11 = ResidualBlock(64)
        self.down2 = UNetDown(64, 128)
        self.ResidualBlock2 = ResidualBlock(128)
        self.ResidualBlock22 = ResidualBlock(128)
        self.down3 = UNetDown(128, 256)
        self.ResidualBlock3 = ResidualBlock(256)
        self.ResidualBlock33 = ResidualBlock(256)
        self.down4 = UNetDown(256, 512)
        self.ResidualBlock4 = ResidualBlock(512)
        self.ResidualBlock44 = ResidualBlock(512)
        self.down5 = UNetDown(512, 512, dropout=drouprate)
        # self.down6 = UNetDown(512, 512, dropout=drouprate)
        # self.down7 = UNetDown(512, 512, dropout=drouprate)
        # self.ResidualBlock7 = ResidualBlock(128)
        # self.ResidualBlock71 = ResidualBlock(128)

        self.up1 = UNetUp(512, 512, dropout=drouprate2)
        self.up2 = UNetUp(1024, 512, dropout=drouprate2)
        self.up3 = UNetUp(768, 256, dropout=drouprate2)
        #self.UPResidualBlock3 = ResidualBlock(1024)
        self.up4 = UNetUp(384, 128)
        #self.UPResidualBlock4 = ResidualBlock(768)
        self.up5 = UNetUp(768, 256)
        self.up6 = UNetUp(384, 64)
        # self.up7 = UNetUp(256, 64)
        self.final1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        self.final2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        self.final3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        self.final4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        self.final5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        self.final6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        self.final7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        self.final8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 7, padding=3),
        )
        # self.final9 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final10 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final11 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final12 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final13 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final14 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final15 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final16 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final17 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final18 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final19 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )
        # self.final20 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, 64, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(64, 32, 5, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 7, padding=3),
        # )

    def forward(self, x, season_feature):
        # U-Net generator with skip connections from encoder to decoder
        x1 = self.channel_attention(x)
        d1 = self.down1(x1, season_feature)
        res11 = self.ResidualBlock1(d1)
        res12 = self.ResidualBlock11(res11) + d1
        d2 = self.down2(res12, season_feature)
        res21 = self.ResidualBlock2(d2)
        res22 = self.ResidualBlock22(res21) + d2
        d3 = self.down3(res22, season_feature)
        res31 = self.ResidualBlock3(d3)
        res32 = self.ResidualBlock33(res31) + d3
        d4 = self.down4(res32, season_feature)
        res41 = self.ResidualBlock4(d4)
        res42 = self.ResidualBlock4(res41) + d4
        d5 = self.down5(res42, season_feature)
        # d6 = self.down6(d5, season_feature)
        # d7 = self.down7(d6, season_feature)

        u1 = self.up1(d5, res42, season_feature)
        u2 = self.up2(u1, d3, season_feature)
        u3 = self.up3(u2, d2, season_feature)
        # resU3 = self.UPResidualBlock3(u3) + u3
        u4 = self.up4(u3, d1, season_feature)
        # resU4 = self.UPResidualBlock4(u4) + u4
        # u5 = self.up5(u4, d2, season_feature)
        # u6 = self.up6(u5, d1, season_feature)
        final1 = self.final1(u4)
        final2 = self.final2(u4)
        # final3 = self.final3(u4)
        # final4 = self.final4(u4)
        # final5 = self.final5(u4)
        # final6 = self.final6(u4)
        # final7 = self.final7(u4)
        # final8 = self.final8(u6)
        # final9 = self.final9(u6)
        # final10 = self.final10(u6)
        # final11 = self.final11(u6)
        # final12 = self.final12(u6)
        # final13 = self.final13(u6)
        # final14 = self.final14(u6)
        # final15 = self.final15(u6)
        # final16 = self.final16(u6)
        # final15 = self.final9(u6)
        # final16 = self.final9(u6)
        # final17 = self.final9(u6)
        # final18 = self.final9(u6)
        # final19 = self.final9(u6)
        u8 = torch.cat((final1, final2), 1)
        return u8


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.norm1 = nn.InstanceNorm2d(in_channels * 4)
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.Conv2d(32, 1, 4, stride=2, padding=1, bias=False)
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        # img_input = self.norm1(img_input)
        out = self.model(img_input)
        return out


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    net = GeneratorUNet()
    print(count_parameters(net))
