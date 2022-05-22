import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch


class SimpleConvolutionEncoder(nn.Module):
    """
    Simple convolution encoder implementation
    """

    def __init__(self):
        super().__init__()

        self.conv2D_1, self.relu_1, self.maxpool_1 = self.convolve_unit(
            in_channels_=3,
            out_channels_=16,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_2, self.relu_2, self.maxpool_2 = self.convolve_unit(
            in_channels_=16,
            out_channels_=32,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_3, self.relu_3, self.maxpool_3 = self.convolve_unit(
            in_channels_=32,
            out_channels_=64,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_4, self.relu_4, self.maxpool_4 = self.convolve_unit(
            in_channels_=64,
            out_channels_=128,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

    @staticmethod
    def convolve_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                      pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, leaky_relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride,
                                            pool_kernel_size, pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, leaky_relu_activation, maxpool_layer

    def forward(self, x):
        # we first downscale the image by repeated convolutions
        x = self.conv2D_1(x)  # 3, 224, 244 ---> 16, 224, 224
        # print('After convolving:', x.size())
        x = self.relu_1(x)
        x = self.maxpool_1(x)  # 16, 224, 224 ---> 16, 222, 222
        # print('After maxpool:', x.size())

        x = self.conv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        # print('After convolving:', x.size())
        x = self.relu_2(x)
        x = self.maxpool_2(x)  # 32, 222, 222 ---> 32, 220, 220
        # print('After maxpool:', x.size())

        x = self.conv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        # print('After convolving:', x.size())
        x = self.relu_3(x)
        x = self.maxpool_3(x)  # 64, 219, 219 ---> 64, 110, 110
        # print('After maxpool:', x.size())

        x = self.conv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        # print('After convolving:', x.size())
        x = self.relu_4(x)
        x = self.maxpool_4(x)  # 128, 108, 108 ---> N, 128, 14, 14
        # print('After maxpool:', x.size())

        # print(f'After linear layer:{x.size()}')
        return x


class SimpleConvolutionEncoderBatchNorm(nn.Module):
    """
    Simple convolution encoder implementation with Batch Normalization
    """

    def __init__(self):
        super().__init__()

        self.conv2D_1, self.batch_norm2D_1, self.relu_1, self.maxpool_1 = self.convolve_batch_norm_unit(
            in_channels_=3,
            out_channels_=16,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_2, self.batch_norm2D_2, self.relu_2, self.maxpool_2 = self.convolve_batch_norm_unit(
            in_channels_=16,
            out_channels_=32,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_3, self.batch_norm2D_3, self.relu_3, self.maxpool_3 = self.convolve_batch_norm_unit(
            in_channels_=32,
            out_channels_=64,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_4, self.batch_norm2D_4, self.relu_4, self.maxpool_4 = self.convolve_batch_norm_unit(
            in_channels_=64,
            out_channels_=128,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.flatten = nn.Flatten()

        # # newly added
        self.linear_1 = nn.Linear(in_features=25088, out_features=1024)
        self.linear_batch_norm1D = nn.BatchNorm1d(num_features=1024, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def convolve_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                      pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, leaky_relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride,
                                            pool_kernel_size, pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, leaky_relu_activation, maxpool_layer

    def forward(self, x):
        # we first downscale the image by repeated convolutions
        x = self.conv2D_1(x)  # 3, 224, 244 ---> 16, 224, 224
        # print('After convolving:', x.size())
        x = self.batch_norm2D_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)  # 16, 224, 224 ---> 16, 222, 222
        # print('After maxpool:', x.size())

        x = self.conv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        # print('After convolving:', x.size())
        x = self.batch_norm2D_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)  # 32, 222, 222 ---> 32, 220, 220
        # print('After maxpool:', x.size())

        x = self.conv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        # print('After convolving:', x.size())
        x = self.batch_norm2D_3(x)
        x = self.relu_3(x)
        x = self.maxpool_3(x)  # 64, 219, 219 ---> 64, 110, 110
        # print('After maxpool:', x.size())

        x = self.conv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        # print('After convolving:', x.size())
        x = self.batch_norm2D_4(x)
        x = self.relu_4(x)
        x = self.maxpool_4(x)  # 128, 108, 108 ---> N, 128, 14, 14
        # print('After maxpool:', x.size())

        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_batch_norm1D(x)
        x = self.relu(x)
        # print(f'After linear layer:{x.size()}')
        return x


class SimpleConvolutionEncoderLR(nn.Module):
    """
    Simple convolution encoder implementation
    """

    def __init__(self):
        super().__init__()

        self.conv2D_1, self.relu_1, self.maxpool_1 = self.convolve_leaky_relu_unit(
            in_channels_=3,
            out_channels_=16,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_2, self.relu_2, self.maxpool_2 = self.convolve_leaky_relu_unit(
            in_channels_=16,
            out_channels_=32,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_3, self.relu_3, self.maxpool_3 = self.convolve_leaky_relu_unit(
            in_channels_=32,
            out_channels_=64,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_4, self.relu_4, self.maxpool_4 = self.convolve_leaky_relu_unit(
            in_channels_=64,
            out_channels_=128,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        # # newly added
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(in_features=25088, out_features=1024)
        self.linear_batch_norm1D = nn.BatchNorm1d(num_features=1024, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def convolve_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                      pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, leaky_relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride,
                                            pool_kernel_size, pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, leaky_relu_activation, maxpool_layer

    def forward(self, x):
        # we first downscale the image by repeated convolutions
        x = self.conv2D_1(x)  # 3, 224, 244 ---> 16, 224, 224
        # print('After convolving:', x.size())
        x = self.maxpool_1(x)
        x = self.relu_1(x)
        # 16, 224, 224 ---> 16, 222, 222
        # print('After maxpool:', x.size())

        x = self.conv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        # print('After convolving:', x.size())
        x = self.maxpool_2(x)  # 32, 222, 222 ---> 32, 220, 220
        x = self.relu_2(x)
        # print('After maxpool:', x.size())

        x = self.conv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        # print('After convolving:', x.size())
        x = self.maxpool_3(x)  # 64, 219, 219 ---> 64, 110, 110
        x = self.relu_3(x)
        # print('After maxpool:', x.size())

        x = self.conv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        # print('After convolving:', x.size())
        x = self.maxpool_4(x)  # 128, 108, 108 ---> N, 128, 14, 14
        x = self.relu_4(x)
        # print('After maxpool:', x.size())

        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_batch_norm1D(x)
        x = self.relu(x)

        # print(f'After linear layer:{x.size()}')
        return x


class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()

        resnet = models.resnet101(pretrained=True)

        for params in resnet.parameters():
            params.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)  # in_feature = 2048
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)

        return x


class Resnet18Encoder(nn.Module):

    def __init__(self):
        super(Resnet18Encoder, self).__init__()

        resnet = models.resnet18(pretrained=True)

        # for params in resnet.parameters():
        #   params.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)  # in_feature = 512
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv
        # print(x.size()) # should return 512

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)  # --> returns 1024 as the final output

        return x

class EfficientNetB0Encoder(nn.Module):
    def __init__(self):
        super(EfficientNetB0Encoder, self).__init__()

        efficientnet_b0 = models.efficientnet_b0(pretrained=True)

        # for params in efficientnet_b0.parameters():
        #     params.requires_grad = False

        modules = list(efficientnet_b0.children())[:-1]
        self.efficientnet_b0 = nn.Sequential(*modules)


        self.fc1 = nn.Linear(list(efficientnet_b0.children())[-1][1].in_features, 1024)  # in_feature = 1280
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.efficientnet_b0(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)

        return x


"""
VAE
"""
class LREncoderVAE(nn.Module):
    """
    Simple convolution encoder implementation
    """

    def __init__(self):
        super().__init__()

        self.conv2D_1, self.relu_1, self.maxpool_1 = self.convolve_leaky_relu_unit(
            in_channels_=3,
            out_channels_=16,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_2, self.relu_2, self.maxpool_2 = self.convolve_leaky_relu_unit(
            in_channels_=16,
            out_channels_=32,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_3, self.relu_3, self.maxpool_3 = self.convolve_leaky_relu_unit(
            in_channels_=32,
            out_channels_=64,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        self.conv2D_4, self.relu_4, self.maxpool_4 = self.convolve_leaky_relu_unit(
            in_channels_=64,
            out_channels_=128,
            conv_kernel_size=(3, 3),
            padding_=(1, 1),
            conv_stride=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2)
        )

        # # newly added
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(in_features=25088, out_features=1024)
        self.linear_batch_norm1D = nn.BatchNorm1d(num_features=1024, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        # Latent vectors mu and sigma
        self.z_mu = nn.Linear(1024, 4096)  # mu = (N, 4096),
        self.z_logvar = nn.Linear(1024, 4096)  # logvar = (N, 4096)

        self.is_training = True

    @staticmethod
    def convolve_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                      pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        relu_activation = nn.ReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride, pool_kernel_size,
                                 pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, leaky_relu_activation, maxpool_layer

    @staticmethod
    def convolve_leaky_relu_batch_norm_unit(in_channels_, out_channels_, conv_kernel_size, padding_, conv_stride,
                                            pool_kernel_size, pool_stride):
        conv2D_layer = nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=conv_kernel_size,
                                 padding=padding_, stride=conv_stride)
        batch_norm_layer = nn.BatchNorm2d(num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
        leaky_relu_activation = nn.LeakyReLU(inplace=True)
        maxpool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        return conv2D_layer, batch_norm_layer, leaky_relu_activation, maxpool_layer

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)  # dimension should be (N, 4096)

        # if self.is_training:
        #     return z
        return z

    def forward(self, x):
        # we first downscale the image by repeated convolutions
        x = self.conv2D_1(x)  # 3, 224, 244 ---> 16, 224, 224
        # print('After convolving:', x.size())
        x = self.maxpool_1(x)
        x = self.relu_1(x)
        # 16, 224, 224 ---> 16, 222, 222
        # print('After maxpool:', x.size())

        x = self.conv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        # print('After convolving:', x.size())
        x = self.maxpool_2(x)  # 32, 222, 222 ---> 32, 220, 220
        x = self.relu_2(x)
        # print('After maxpool:', x.size())

        x = self.conv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        # print('After convolving:', x.size())
        x = self.maxpool_3(x)  # 64, 219, 219 ---> 64, 110, 110
        x = self.relu_3(x)
        # print('After maxpool:', x.size())

        x = self.conv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        # print('After convolving:', x.size())
        x = self.maxpool_4(x)  # 128, 108, 108 ---> N, 128, 14, 14
        x = self.relu_4(x)
        # print('After maxpool:', x.size())

        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_batch_norm1D(x)
        x = self.relu(x)

        # x = F.dropout(x, p=self.drop_p, training=self.training)
        z_mean, z_logvar = self.z_mu(x), self.z_logvar(x)  # size of both (N, 4096)

        encoded = self.reparameterize(z_mu=z_mean, z_log_var=z_logvar)  # (N, 4096)

        # print(f'After linear layer:{x.size()}')
        return encoded, z_mean, z_logvar


class Resnet101VaeEncoder(nn.Module):
    def __init__(self):
        super(Resnet101VaeEncoder, self).__init__()

        resnet = models.resnet101(pretrained=True)

        for params in resnet.parameters():
            params.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)  # in_feature = 2048
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        # Latent vectors mu and sigma
        self.z_mu = nn.Linear(768, 768)  # mu = (N, 768),
        self.z_logvar = nn.Linear(768, 768)  # logvar = (N, 768)

        self.is_training = True

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)  # dimension should be (N, 768)

        if self.is_training:
            return z

        return z_mu

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)

        # x = F.dropout(x, p=self.drop_p, training=self.training)
        z_mean, z_logvar = self.z_mu(x), self.z_logvar(x)  # size of both (N, 768)

        encoded = self.reparameterize(z_mu=z_mean, z_log_var=z_logvar)  # (N, 768)
        # print('vae101encoder : ', encoded.size())

        return encoded, z_mean, z_logvar


class Resnet50VaeEncoder(nn.Module):
    def __init__(self):
        super(Resnet50VaeEncoder, self).__init__()

        resnet = models.resnet50(pretrained=True)

        for params in resnet.parameters():
            params.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)  # in_feature = 2048
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        # Latent vectors mu and sigma
        self.z_mu = nn.Linear(768, 768)  # mu = (N, 768),
        self.z_logvar = nn.Linear(768, 768)  # logvar = (N, 768)

        self.is_training = True

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)  # dimension should be (N, 768)

        if self.is_training:
            return z
        return z_mu

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)

        # x = F.dropout(x, p=self.drop_p, training=self.training)
        z_mean, z_logvar = self.z_mu(x), self.z_logvar(x)  # size of both (N, 768)

        encoded = self.reparameterize(z_mu=z_mean, z_log_var=z_logvar)  # (N, 768)
        # print('vae101encoder : ', encoded.size())

        return encoded, z_mean, z_logvar


class EfficientNetB0VaeEncoder(nn.Module):
    def __init__(self):
        super(EfficientNetB0VaeEncoder, self).__init__()

        efficientnet_b0 = models.efficientnet_b0(pretrained=True)

        for params in efficientnet_b0.parameters():
            params.requires_grad = False

        modules = list(efficientnet_b0.children())[:-1]
        self.efficientnet_b0 = nn.Sequential(*modules)


        self.fc1 = nn.Linear(list(efficientnet_b0.children())[-1][1].in_features, 1024)  # in_feature = 1280
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        # Latent vectors mu and sigma
        self.z_mu = nn.Linear(768, 768)  # mu = (N, 768),
        self.z_logvar = nn.Linear(768, 768)  # logvar = (N, 768)
        
        self.is_training = True

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)  # dimension should be (N, 768)
        
        if self.is_training:
            return z
        
        return z_mu

    def forward(self, x):
        x = self.efficientnet_b0(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)

        # x = F.dropout(x, p=self.drop_p, training=self.training)
        z_mean, z_logvar = self.z_mu(x), self.z_logvar(x)  # size of both (N, 768)

        encoded = self.reparameterize(z_mu=z_mean, z_log_var=z_logvar)  # (N, 768)
        # print('vae101encoder : ', encoded.size())

        return encoded, z_mean, z_logvar


class Resnet18FullVaeEncoder(nn.Module):
    def __init__(self):
        super(Resnet18FullVaeEncoder, self).__init__()

        resnet = models.resnet18(pretrained=True)

        # for params in resnet.parameters():
        #     params.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)  # in_feature = 512
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        # Latent vectors mu and sigma
        self.z_mu = nn.Linear(1024, 1024)  # mu = (N, 4096),
        self.z_logvar = nn.Linear(1024, 1024)  # logvar = (N, 4096)

        self.is_training = True

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)  # dimension should be (N, 4096)

        if self.is_training:
            return z
        
        return z_mu

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)

        # x = F.dropout(x, p=self.drop_p, training=self.training)
        z_mean, z_logvar = self.z_mu(x), self.z_logvar(x)  # size of both (N, 4096)

        encoded = self.reparameterize(z_mu=z_mean, z_log_var=z_logvar)  # (N, 4096)
        # print('vae101encoder : ', encoded.size())

        return encoded, z_mean, z_logvar