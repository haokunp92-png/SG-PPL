import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import conv, deconv, predict_mask


class SpixelNet(nn.Module):
    def __init__(self, batchNorm=True):
        super(SpixelNet, self).__init__()
        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)
        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)
        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)
        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
        self.pred_mask0 = predict_mask(16, self.assign_ch)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x))
        out2 = self.conv1b(self.conv1a(out1))
        out3 = self.conv2b(self.conv2a(out2))
        out4 = self.conv3b(self.conv3a(out3))
        out5 = self.conv4b(self.conv4a(out4))

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        mask0 = self.pred_mask0(self.conv0_1(concat0))
        return self.softmax(mask0)

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if 'weight' in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


def SpixelNet1l(data=None):
    model = SpixelNet(batchNorm=False)
    if data is not None and 'state_dict' in data:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


def SpixelNet1l_bn(data=None):
    model = SpixelNet(batchNorm=True)
    if data is not None and 'state_dict' in data:
        model.load_state_dict(data['state_dict'], strict=False)
    return model
