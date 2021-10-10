import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        utils.initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x #Elementwise Sum
 

class generator(nn.Module):
    # initializers
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=6):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3), #k7n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1), #k3n128s2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1), #k3n256s1
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), #k3n256s1
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            # nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1), #k3n128s1/2
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            # nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1), #k3n64s1/2
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nf * 2, nf, 3, 1, 1), #k3n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3), #k7n3s1
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)

        return output


# class ResBlock(nn.Module):
#     def __init__(self, num_channel):
#         super(ResBlock, self).__init__()
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(num_channel, num_channel, 3, 1, 1),
#             nn.BatchNorm2d(num_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_channel, num_channel, 3, 1, 1),
#             nn.BatchNorm2d(num_channel))
#         self.activation = nn.ReLU(inplace=True)

#     def forward(self, inputs):
#         output = self.conv_layer(inputs)
#         output = self.activation(output + inputs)
#         return output


# class DownBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(DownBlock, self).__init__()
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, 2, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True))


#     def forward(self, inputs):
#         output = self.conv_layer(inputs)
#         return output


# class UpBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, is_last=False):
#         super(UpBlock, self).__init__()
#         self.is_last = is_last
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, 3, 1, 1),
#             nn.BatchNorm2d(in_channel),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channel, out_channel, 3, 1, 1))
#         self.act = nn.Sequential(
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True))
#         self.last_act = nn.Tanh()


#     def forward(self, inputs):
#         output = self.conv_layer(inputs)
#         if self.is_last:
#             output = self.last_act(output)
#         else:
#             output = self.act(output)
#         return output



# class generator(nn.Module):
#     def __init__(self, in_nc, out_nc, nf=32, nb=6):
#         super(generator, self).__init__()
#         num_channel=32
#         num_blocks=4
#         self.down1 = DownBlock(3, num_channel)
#         self.down2 = DownBlock(num_channel, num_channel*2)
#         self.down3 = DownBlock(num_channel*2, num_channel*3)
#         self.down4 = DownBlock(num_channel*3, num_channel*4)
#         res_blocks = [ResBlock(num_channel*4)]*num_blocks
#         self.res_blocks = nn.Sequential(*res_blocks)
#         self.up1 = UpBlock(num_channel*4, num_channel*3)
#         self.up2 = UpBlock(num_channel*3, num_channel*2)
#         self.up3 = UpBlock(num_channel*2, num_channel)
#         self.up4 = UpBlock(num_channel, 3, is_last=True)

#     def forward(self, inputs):
#         down1 = self.down1(inputs)
#         down2 = self.down2(down1)
#         down3 = self.down3(down2)
#         down4 = self.down4(down3)
#         down4 = self.res_blocks(down4)
#         up1 = self.up1(down4)
#         up2 = self.up2(up1+down3)
#         up3 = self.up3(up2+down2)
#         up4 = self.up4(up3+down1)
#         return up4


class discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        # input = torch.cat((input1, input2), 1)
        output = self.convs(input)

        return output


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x


class RestNet18(nn.Module):
    def __init__(self, init_weights=None, num_classes=2):
        super(RestNet18, self).__init__()
        self.init_weights = init_weights
        self.num_classes = num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        self.fc = torch.nn.Linear(25088, self.num_classes)

        if self.init_weights is not None:
            print('Load Resnet18 pretrained')
            self.model = torch.load(self.init_weights)

        self.features = nn.Sequential(
            *list(self.model.children())[:-3]
            )

    def forward(self, x):
        x = self.features(x)

        return x
