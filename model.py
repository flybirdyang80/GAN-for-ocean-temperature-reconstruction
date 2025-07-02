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
        # CBAM代替卷积
        self.CABAMDown = CBAM(out_size * 2)
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
        # y2 = self.seasonDown(y1, season_feature)
        # y3 = torch.cat((y1, y2), 1)  # 先并起来
        # y3CBAM = self.CABAMDown(y3)
        # y4 = self.conv1(y3CBAM)  # 季节特征和Unet输出并起来再恢复原来的通道
        # y4 = self.norm1(y4)
        # y5 = self.leakyrelu(y4)
        return y1


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        self.seasonUp = ChannelSupervisionUnitUP(out_size, out_size)
        self.CABAMUP = CBAM(out_size * 2)
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
        # y2 = self.seasonUp(y1, season_feature)  # 提取季节特征
        # y3 = torch.cat((y1, y2), 1)  # 先并起来
        # y3CBAM = self.CABAMUP(y3)
        # y4 = self.conv1(y3CBAM)  # 季节特征和Unet输出并起来再恢复原来的通道
        # y4 = self.norm1(y4)
        # y5 = F.relu(y4)
        y7 = torch.cat((y1, skip_input), 1)
        return y7


class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class ChannelAttention_input(nn.Module):
    def __init__(self, in_channels, ratio=1):
        super(ChannelAttention_input, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class CBAM(nn.Module):
    """
    CBAM混合注意力机制
    """

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.bn1 = nn.BatchNorm2d(channels)  # 批归一化
        self.bn2 = nn.BatchNorm2d(channels)
        self.droupout = nn.Dropout(0.001)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=7, padding=3)

    def forward(self, x):
        y1 = self.leakyrelu(self.bn1(self.conv1(x)))
        y2 = self.bn1(self.conv2(y1))
        y3 = self.leakyrelu(y2 + x)
        y4 = self.droupout(y3)
        return y4


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


##############################
# LSEnet的季节特征提取模块(加到编码器里)
##############################
class ChannelSupervisionUnit(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelSupervisionUnit, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channel, 24)
        self.fc2 = nn.Linear(24 + 12, out_channel)  # 假设 season_feature 是3维的
        self.channel_wise_multiply = ChannelWiseMultiply()
        self.leakyrelu = nn.LeakyReLU(0.06)

    def forward(self, x, season_feature):
        s = x
        o = x
        s = self.global_avg_pool(s)
        s = s.view(s.size(0), -1)  # 展平
        s = self.fc1(s)
        s = self.leakyrelu(s)
        s = torch.cat([s, season_feature], dim=1)
        s = self.fc2(s)
        s = self.leakyrelu(s)
        o = self.channel_wise_multiply(o, s)
        return o


class ChannelWiseMultiply(nn.Module):
    def __init__(self):
        super(ChannelWiseMultiply, self).__init__()

    def forward(self, o, s):
        height = int(o.shape[2])
        width = int(o.shape[3])
        s = s.unsqueeze(2).unsqueeze(3)  # (bs, classes, 1, 1)
        x = s.repeat(1, 1, height, width)  # (bs, classes, h, w)
        # res = o * x
        res = x
        return res


class ChannelSupervisionUnitUP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelSupervisionUnitUP, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channel, 24)
        self.fc2 = nn.Linear(24 + 12, out_channel)  # 假设 season_feature 是3维的
        self.channel_wise_multiply = ChannelWiseMultiplyUP()
        self.leakyrelu = nn.LeakyReLU(0.06)

    def forward(self, x, season_feature):
        s = x
        o = x
        s = self.global_avg_pool(s)
        s = s.view(s.size(0), -1)  # 展平
        s = self.fc1(s)
        s = self.leakyrelu(s)
        s = torch.cat([s, season_feature], dim=1)
        s = self.fc2(s)
        s = self.leakyrelu(s)
        o = self.channel_wise_multiply(o, s)
        return o


class ChannelWiseMultiplyUP(nn.Module):
    def __init__(self):
        super(ChannelWiseMultiplyUP, self).__init__()

    def forward(self, o, s):
        height = o.shape[2]
        width = o.shape[3]
        s = s.unsqueeze(2).unsqueeze(3)  # (bs, classes, 1, 1)
        x = s.repeat(1, 1, height, width)  # (bs, classes, h, w)
        # res = o * x
        res = x
        return res


class Self_Attn(nn.Module):
    """ Self attention Layer for image patches"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.patch_size = 8
        self.channel_in = in_dim
        # Define convolutions for query, key, and value
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.conv1d = nn.Conv2d(in_channels=in_dim // 8, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Unfold the input tensor into patches
        proj_query = proj_query.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height // (self.patch_size ** 2)).permute(0,
                                                                                                                     2,
                                                                                                                     1)

        proj_key = proj_key.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height // (self.patch_size ** 2))

        d_k = proj_key.size(-1)
        energy = torch.bmm(proj_query, proj_key) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float, device=proj_query.device))

        attention = self.softmax(energy)

        proj_value = proj_value.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height // (self.patch_size ** 2))

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.permute(0, 2, 1).contiguous().view(m_batchsize, C // 8, width, height)
        out = self.conv1d(out)
        out = self.gamma * out + x
        return out


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super(GeneratorUNet, self).__init__()
        drouprate = 0.1
        drouprate2 = 0.1
        self.channel_attention = ChannelAttention_input(in_channels)
        self.down1 = UNetDown(in_channels, 64, normalize=True)
        self.attnD1 = Self_Attn(64)
        self.ResidualBlock1 = ResidualBlock(64)
        self.ResidualBlock11 = ResidualBlock(64)
        self.down2 = UNetDown(64, 128)
        self.attnD2 = Self_Attn(128)
        self.ResidualBlock2 = ResidualBlock(128)
        self.ResidualBlock22 = ResidualBlock(128)
        self.down3 = UNetDown(128, 256)
        self.attnD3 = Self_Attn(256)
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
        # at_d1 = self.attnD1(d1)
        res11 = self.ResidualBlock1(d1)
        res12 = self.ResidualBlock11(res11) + d1
        d2 = self.down2(res12, season_feature)
        # at_d2 = self.attnD2(d2)
        res21 = self.ResidualBlock2(d2)
        res22 = self.ResidualBlock22(res21) + d2
        d3 = self.down3(res22, season_feature)
        # at_d3 = self.attnD3(d3)
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
        # final20 = self.final9(u6)
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
