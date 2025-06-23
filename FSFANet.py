import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Unet3plus.init_weights import init_weights
from model.cxlNet.component import (Res_CoordAtt_block, STEM, Res_block, Res_CBAM_block, InvertedResidual,
                                    STEM_wo_PAM,STEM_wo_MDCM)
from model.MCANet.MCANet import MSCA,MSCA_0120,MSCA_0125,ASCA_0229
import time
import numbers
from einops import rearrange


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, kernel):
        super(ConvLayer, self).__init__()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=self.weight.data.shape[2] // 2)


class PAM(nn.Module):
    def __init__(self):
        super(PAM, self).__init__()

        # 定义不同尺度的卷积层
        self.conv_d3_15 = ConvLayer(torch.FloatTensor([[-1, 0, 0],
                                                       [0, 2, 0],
                                                       [0, 0, -1]]).unsqueeze(0).unsqueeze(0))
        self.conv_d3_26 = ConvLayer(torch.FloatTensor([[0, -1, 0],
                                                       [0, 2, 0],
                                                       [0, -1, 0]]).unsqueeze(0).unsqueeze(0))
        self.conv_d3_37 = ConvLayer(torch.FloatTensor([[0, 0, -1],
                                                       [0, 2, 0],
                                                       [-1, 0, 0]]).unsqueeze(0).unsqueeze(0))
        self.conv_d3_48 = ConvLayer(torch.FloatTensor([[0, 0, 0],
                                                       [-1, 2, -1],
                                                       [0, 0, 0]]).unsqueeze(0).unsqueeze(0))

        self.conv_d5_15 = ConvLayer(torch.FloatTensor(
            [[-1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, -1]]).unsqueeze(0).unsqueeze(0))
        self.conv_d5_26 = ConvLayer(torch.FloatTensor(
            [[0, 0, -1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, -1, 0, 0]]).unsqueeze(0).unsqueeze(0))
        self.conv_d5_37 = ConvLayer(torch.FloatTensor(
            [[0, 0, 0, 0, -1],
             [0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 0, 0],
             [-1, 0, 0, 0, 0]]).unsqueeze(0).unsqueeze(0))
        self.conv_d5_48 = ConvLayer(torch.FloatTensor(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [-1, 0, 2, 0, -1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]).unsqueeze(0).unsqueeze(0))

        self.conv_d7_15 = ConvLayer(torch.FloatTensor(
            [[-1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, -1]]).unsqueeze(0).unsqueeze(0))
        self.conv_d7_26 = ConvLayer(torch.FloatTensor(
            [[0, 0, 0, -1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, -1, 0, 0, 0]]).unsqueeze(0).unsqueeze(0))
        self.conv_d7_37 = ConvLayer(torch.FloatTensor(
            [[0, 0, 0, 0, 0, 0, -1],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [-1, 0, 0, 0, 0, 0, 0]]).unsqueeze(0).unsqueeze(0))
        self.conv_d7_48 = ConvLayer(torch.FloatTensor(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [-1, 0, 0, 2, 0, 0, -1],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]]).unsqueeze(0).unsqueeze(0))

        # 其他尺度的卷积层类似地定义

    def forward(self, x):
        # 进行卷积操作
        d3 = self.conv_d3_15(x) + self.conv_d3_26(x) + self.conv_d3_37(x) + self.conv_d3_48(x)
        d5 = self.conv_d5_15(x) + self.conv_d5_26(x) + self.conv_d5_37(x) + self.conv_d5_48(x)
        d7 = self.conv_d7_15(x) + self.conv_d7_26(x) + self.conv_d7_37(x) + self.conv_d7_48(x)

        dmax = torch.max(d3, d5)
        dmax = torch.max(dmax, d7)
        dmax = torch.sigmoid(torch.relu(dmax))

        return dmax


class STEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STEM, self).__init__()
        assert out_channels % 2 == 0, '输出通道数应为偶数'
        self.resBlock1 = Res_block(in_channels, in_channels)
        self.resBlock2 = Res_block(in_channels, max(1, out_channels // 2))
        self.pam = PAM()
        self.mdcm = MDCM(in_channels, max(1, out_channels // 2))

    def forward(self, x):
        x_m2am = self.pam(x)
        resBlock1_out = self.resBlock1(x)
        assert x_m2am.shape == resBlock1_out.shape
        x_m2am = torch.mul(x_m2am, resBlock1_out)
        resBlock1_out = torch.add(x_m2am, resBlock1_out)
        out = self.resBlock2(resBlock1_out)
        x_mdcm = self.mdcm(x)

        return torch.cat((out, x_mdcm), dim=1)

class MDCM(nn.Module):
    """
    Muti Dilation Convolution Module
    """

    def __init__(self, in_channels, out_channels):
        super(MDCM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, max(1, out_channels // 2), kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(max(1, out_channels // 2), max(1, out_channels // 2), kernel_size=1)
        self.conv2_1 = nn.Conv2d(max(1, out_channels // 2), out_channels, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(max(1, out_channels // 2), out_channels, kernel_size=3, dilation=2, padding=2)
        self.conv2_3 = nn.Conv2d(max(1, out_channels // 2), out_channels, kernel_size=3, dilation=4, padding=4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        return x1 + x2 + x3

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ASCA_0229(nn.Module):

    def __init__(self, dim, out_channels, num_heads, LayerNorm_type = 'BiasFree',):
        super().__init__()
        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_input = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_output = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = LayerNorm(out_channels, LayerNorm_type)
        # self.norm1 = nn.LayerNorm()
        self.project_out1q = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.project_out2q = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.project_out1v = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.project_out2v = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.project_out1k = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.project_out2k = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv0_1 = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1), groups=out_channels)  # 为什么要分组卷积呢
        self.conv0_2 = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0), groups=out_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, (1, 7), padding=(0, 3), groups=out_channels)
        self.conv1_2 = nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), groups=out_channels)
        self.conv2_1 = nn.Conv2d(
            out_channels, out_channels, (1, 11), padding=(0, 5), groups=out_channels)
        self.conv2_2 = nn.Conv2d(
            out_channels, out_channels, (11, 1), padding=(5, 0), groups=out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_input(x)
        x1 = self.norm1(x)
        attn_00 = self.conv0_1(x1)
        attn_01 = self.conv0_2(x1)
        attn_10 = self.conv1_1(x1)
        attn_11 = self.conv1_2(x1)
        attn_20 = self.conv2_1(x1)
        attn_21 = self.conv2_2(x1)
        out1 = attn_00 + attn_10 + attn_20
        out2 = attn_01 + attn_11 + attn_21
        out1k = self.project_out1k(out1)
        out2k = self.project_out2k(out2)
        out1v = self.project_out1v(out1)
        out2v = self.project_out2v(out2)
        out1q = self.project_out1q(out1)
        out2q = self.project_out2q(out2)
        k1 = rearrange(out1k, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1v, 'b (head c) h w -> b head h (w c)', head=self.num_heads)

        k2 = rearrange(out2k, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2v, 'b (head c) h w -> b head w (h c)', head=self.num_heads)

        self_q1 = rearrange(out1q, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        cross_q1 = rearrange(out1q, 'b (head c) h w -> b head w (h c)', head=self.num_heads) # 给k2 v2使用

        self_q2 = rearrange(out2q, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        cross_q2 = rearrange(out2q, 'b (head c) h w -> b head h (w c)', head=self.num_heads)  # 给k1 v1使用

        self_q1 = torch.nn.functional.normalize(self_q1, dim=-1)
        cross_q1 = torch.nn.functional.normalize(cross_q1, dim=-1)
        self_q2 = torch.nn.functional.normalize(self_q2, dim=-1)
        cross_q2 = torch.nn.functional.normalize(cross_q2, dim=-1)

        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        self_attn1 = (self_q1 @ k1.transpose(-2, -1))
        self_attn1 = self_attn1.softmax(dim=-1)
        self_attn1_out = (self_attn1 @ v1) + self_q1

        self_attn2 = (self_q2 @ k2.transpose(-2, -1))
        self_attn2 = self_attn2.softmax(dim=-1)
        self_attn2_out = (self_attn2 @ v2) + self_q2

        cross_attn1 = (cross_q2 @ k1.transpose(-2, -1))
        cross_attn1 = cross_attn1.softmax(dim=-1)
        cross_attn1_out = (cross_attn1 @ v1) + cross_q2

        cross_attn2 = (cross_q1 @ k2.transpose(-2, -1))
        cross_attn2 = cross_attn2.softmax(dim=-1)
        cross_attn2_out = (cross_attn2 @ v2) + cross_q1

        self_attn1_out = rearrange(self_attn1_out, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        self_attn2_out = rearrange(self_attn2_out, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        cross_attn1_out = rearrange(cross_attn1_out, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        cross_attn2_out = rearrange(cross_attn2_out, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self_attn1_out + self_attn2_out + cross_attn1_out + cross_attn2_out + x
        out = self.conv_output(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out


class _Head(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0):
        super(_Head, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

class FSFANet(nn.Module):

    def __init__(self, num_classes=1, input_channels=1, block=Res_block, num_blocks=[2, 2, 2, 2],
                 nb_filter=[32, 64, 64, 128], decode_filter=[64, 64, 32]):
        super().__init__()
        STEM_out_channels = 8
        self.stem = STEM(in_channels=input_channels, out_channels=STEM_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.down_2 = nn.MaxPool2d(2, 2)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down_4 = nn.MaxPool2d(4, 4)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.convStemtoE1 = self._make_layer(block, STEM_out_channels, nb_filter[0])
        self.convE1toE2 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])  # 320*320
        self.convE2toE3 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.convE3toE4 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        # 解码器部分
        self.convD3 = self._make_layer(block,decode_filter[0],decode_filter[0],1)
        self.convD2 = self._make_layer(block,decode_filter[1],decode_filter[1],1)

        self.asca_toD3 = ASCA_0229(nb_filter[0] + nb_filter[1] + nb_filter[2] + nb_filter[3],
                              out_channels=decode_filter[0], num_heads=8)
        self.asca_toD2 = ASCA_0229(nb_filter[0] + nb_filter[1] + nb_filter[3] + decode_filter[0],
                              out_channels=decode_filter[1], num_heads=8)

        self.convD1 = self._make_layer(block, nb_filter[0] + nb_filter[3] + decode_filter[0] + decode_filter[1], decode_filter[2], 1)

        self.final = _Head(decode_filter[-1],num_classes)
    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        E1 = self.convStemtoE1(self.stem(input))
        E2 = self.convE1toE2(self.down_2(E1))
        E3 = self.convE2toE3(self.down_2(E2))
        E4 = self.convE3toE4(self.down_2(E3))

        D3 = self.asca_toD3(torch.cat([self.down_4(E1),self.down_2(E2),E3,self.up_2(E4)], dim=1))
        D3 = self.convD3(D3)
        D2 = self.asca_toD2(torch.cat([self.down_2(E1),E2,self.up_2(D3),self.up_4(E4)], dim=1))
        D2 = self.convD2(D2)
        D1 = self.convD1(torch.cat([E1,self.up_2(D2),self.up_4(D3),self.up_8(E4)], dim=1))
        output = self.final(D1)
        return output


if __name__ == '__main__':
    model = FSFANet()

    x = torch.rand((1, 1, 256, 256))
    out = model(x)





