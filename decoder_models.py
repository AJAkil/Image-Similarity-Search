import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class SimpleConvolutionDecoder(nn.Module):
    """
    Simple convolution decoder implementation
    """

    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=1024, out_features=25088)
        self.linear_batch_norm1D = nn.BatchNorm1d(
            num_features=25088, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.reshape = Reshape(-1, 128, 14, 14)

        self.deconv2D_1,  self.relu_1 = self.deconvolve_unit(
            in_channels_=128,
            out_channels_=64,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_2,  self.relu_2 = self.deconvolve_unit(
            in_channels_=64,
            out_channels_=32,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_3,  self.relu_3 = self.deconvolve_unit(
            in_channels_=32,
            out_channels_=16,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_4,  self.relu_4 = self.deconvolve_unit(
            in_channels_=16,
            out_channels_=3,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

    @staticmethod
    def deconvolve_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, relu_activation

    @staticmethod
    def deconvolve_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        leaky_relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, leaky_relu_activation

    @staticmethod
    def deconvolve_batch_norm_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        batch_norm_layer = nn.BatchNorm2d(
            num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, batch_norm_layer, relu_activation

    @staticmethod
    def deconvolve_batch_norm_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        batch_norm_layer = nn.BatchNorm2d(
            num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, batch_norm_layer, relu_activation

    def forward(self, x):
        # we first downscale the image by repeated convolutions
        # print('Before Deconvolution: ', x.size())
        x = self.linear_1(x)
        # print('here')
        x = self.linear_batch_norm1D(x)
        x = self.relu(x)

        x = self.reshape(x)

        # print(f'Reshaping to:{x.size()}')

        x = self.deconv2D_1(x)
        #print('After deconvolving:', x.size())
        x = self.relu_1(x)

        x = self.deconv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        #print('After deconvolving:', x.size())
        x = self.relu_2(x)

        x = self.deconv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        #print('After deconvolving:', x.size())
        x = self.relu_3(x)

        x = self.deconv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        #print('After deconvolving:', x.size())
        x = self.relu_4(x)

        return x


class SimpleConvolutionDecoderBatchNorm(nn.Module):
    """
    Simple convolution decoder implementation
    """

    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=1024, out_features=25088)
        self.linear_batch_norm1D = nn.BatchNorm1d(
            num_features=25088, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.reshape = Reshape(-1, 128, 14, 14)

        self.deconv2D_1, self.batch_norm2D_1, self.relu_1 = self.deconvolve_batch_norm_unit(
            in_channels_=128,
            out_channels_=64,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_2, self.batch_norm2D_2, self.relu_2 = self.deconvolve_batch_norm_unit(
            in_channels_=64,
            out_channels_=32,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_3, self.batch_norm2D_3, self.relu_3 = self.deconvolve_batch_norm_unit(
            in_channels_=32,
            out_channels_=16,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_4, self.batch_norm2D_4, self.relu_4 = self.deconvolve_batch_norm_unit(
            in_channels_=16,
            out_channels_=3,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

    @staticmethod
    def deconvolve_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, relu_activation

    @staticmethod
    def deconvolve_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        leaky_relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, leaky_relu_activation

    @staticmethod
    def deconvolve_batch_norm_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        batch_norm_layer = nn.BatchNorm2d(
            num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, batch_norm_layer, relu_activation

    @staticmethod
    def deconvolve_batch_norm_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        batch_norm_layer = nn.BatchNorm2d(
            num_features=out_channels_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, batch_norm_layer, relu_activation

    def forward(self, x):
        # we first downscale the image by repeated convolutions
        # print('Before Deconvolution: ', x.size())
        x = self.linear_1(x)
        # print('here')
        x = self.linear_batch_norm1D(x)
        x = self.relu(x)

        x = self.reshape(x)

        # print(f'Reshaping to:{x.size()}')

        x = self.deconv2D_1(x)
        x = self.batch_norm2D_1(x)
        #print('After deconvolving:', x.size())
        x = self.relu_1(x)

        x = self.deconv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        #print('After deconvolving:', x.size())
        x = self.batch_norm2D_2(x)
        x = self.relu_2(x)

        x = self.deconv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        #print('After deconvolving:', x.size())
        x = self.batch_norm2D_3(x)
        x = self.relu_3(x)

        x = self.deconv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        x = self.batch_norm2D_4(x)
        #print('After deconvolving:', x.size())
        x = self.relu_4(x)

        return x


class SimpleConvolutionDecoderLR(nn.Module):
    """
    Simple convolution decoder implementation
    """

    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=1024, out_features=25088)
        self.linear_batch_norm1D = nn.BatchNorm1d(
            num_features=25088, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.reshape = Reshape(-1, 128, 14, 14)

        self.deconv2D_1, self.relu_1 = self.deconvolve_leaky_relu_unit(
            in_channels_=128,
            out_channels_=64,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_2, self.relu_2 = self.deconvolve_leaky_relu_unit(
            in_channels_=64,
            out_channels_=32,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_3, self.relu_3 = self.deconvolve_leaky_relu_unit(
            in_channels_=32,
            out_channels_=16,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_4, self.relu_4 = self.deconvolve_leaky_relu_unit(
            in_channels_=16,
            out_channels_=3,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

    @staticmethod
    def deconvolve_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, relu_activation

    @staticmethod
    def deconvolve_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        leaky_relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, leaky_relu_activation

    def forward(self, x):

        x = self.linear_1(x)
        x = self.linear_batch_norm1D(x)
        x = self.relu(x)

        x = self.reshape(x)
        # we first downscale the image by repeated convolutions
        #print('Before Deconvolution: ', x.size())
        x = self.deconv2D_1(x)
        #print('After deconvolving:', x.size())
        x = self.relu_1(x)

        x = self.deconv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        #print('After deconvolving:', x.size())
        x = self.relu_2(x)

        x = self.deconv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        #print('After deconvolving:', x.size())
        x = self.relu_3(x)

        x = self.deconv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        #print('After deconvolving:', x.size())
        x = self.relu_4(x)

        return x

class LRDecoderVAE(nn.Module):
    """
    Simple convolution decoder implementation
    """

    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=4096, out_features=1024)
        self.linear_1_batch_norm1D = nn.BatchNorm1d(
            num_features=1024, momentum=0.01)

        self.linear_2 = nn.Linear(in_features=1024, out_features=25088)
        self.linear_2_batch_norm1D = nn.BatchNorm1d(
            num_features=25088, momentum=0.01)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.reshape = Reshape(-1, 128, 14, 14)

        self.deconv2D_1, self.relu_1 = self.deconvolve_leaky_relu_unit(
            in_channels_=128,
            out_channels_=64,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_2, self.relu_2 = self.deconvolve_leaky_relu_unit(
            in_channels_=64,
            out_channels_=32,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_3, self.relu_3 = self.deconvolve_leaky_relu_unit(
            in_channels_=32,
            out_channels_=16,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_4, self.relu_4 = self.deconvolve_leaky_relu_unit(
            in_channels_=16,
            out_channels_=3,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

    @staticmethod
    def deconvolve_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, relu_activation

    @staticmethod
    def deconvolve_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        leaky_relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, leaky_relu_activation

    def forward(self, x):

        x = self.linear_1(x)
        x = self.linear_1_batch_norm1D(x)
        x = self.relu(x)

        x = self.linear_2(x)
        x = self.linear_2_batch_norm1D(x)
        x = self.relu(x)

        x = self.reshape(x)
        # we first downscale the image by repeated convolutions
        #print('Before Deconvolution: ', x.size())
        x = self.deconv2D_1(x)
        #print('After deconvolving:', x.size())
        x = self.relu_1(x)

        x = self.deconv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        #print('After deconvolving:', x.size())
        x = self.relu_2(x)

        x = self.deconv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        #print('After deconvolving:', x.size())
        x = self.relu_3(x)

        x = self.deconv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        #print('After deconvolving:', x.size())
        x = self.sigmoid(x)

        return x

class SimpleConvolutionDecoderResnet(nn.Module):
    """
    Simple convolution decoder implementation
    """

    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=768, out_features=25088)
        self.linear_batch_norm1D = nn.BatchNorm1d(
            num_features=25088, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.reshape = Reshape(-1, 128, 14, 14)

        self.deconv2D_1, self.relu_1 = self.deconvolve_unit(
            in_channels_=128,
            out_channels_=64,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_2, self.relu_2 = self.deconvolve_unit(
            in_channels_=64,
            out_channels_=32,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_3, self.relu_3 = self.deconvolve_unit(
            in_channels_=32,
            out_channels_=16,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_4, self.relu_4 = self.deconvolve_unit(
            in_channels_=16,
            out_channels_=3,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

    @staticmethod
    def deconvolve_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, relu_activation

    @staticmethod
    def deconvolve_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        leaky_relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, leaky_relu_activation

    def forward(self, x):

        x = self.linear_1(x)
        # print('here')
        x = self.linear_batch_norm1D(x)
        x = self.relu(x)

        x = self.reshape(x)
        # we first downscale the image by repeated convolutions
        #print('Before Deconvolution: ', x.size())
        x = self.deconv2D_1(x)
        #print('After deconvolving:', x.size())
        x = self.relu_1(x)

        x = self.deconv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        #print('After deconvolving:', x.size())
        x = self.relu_2(x)

        x = self.deconv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        #print('After deconvolving:', x.size())
        x = self.relu_3(x)

        x = self.deconv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        #print('After deconvolving:', x.size())
        x = self.relu_4(x)

        return x


class SimpleConvolutionDecoderResnetVAE(nn.Module):
    """
    Simple convolution decoder implementation
    """

    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=768, out_features=1024)
        self.linear_1_batch_norm1D = nn.BatchNorm1d(
            num_features=1024, momentum=0.01)

        self.linear_2 = nn.Linear(in_features=1024, out_features=25088)
        self.linear_2_batch_norm1D = nn.BatchNorm1d(
            num_features=25088, momentum=0.01)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.reshape = Reshape(-1, 128, 14, 14)

        self.deconv2D_1, self.batch_norm2D_1, self.relu_1 = self.deconvolve_batch_norm_unit(
            in_channels_=128,
            out_channels_=64,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_2, self.batch_norm2D_2, self.relu_2 = self.deconvolve_batch_norm_unit(
            in_channels_=64,
            out_channels_=32,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_3, self.batch_norm2D_3, self.relu_3 = self.deconvolve_batch_norm_unit(
            in_channels_=32,
            out_channels_=16,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_4, self.batch_norm2D_4, self.relu_4 = self.deconvolve_batch_norm_unit(
            in_channels_=16,
            out_channels_=3,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

    @staticmethod
    def deconvolve_batch_norm_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        batch_norm_layer = nn.BatchNorm2d(
            num_features=out_channels_, momentum=0.01)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, batch_norm_layer, relu_activation

    @staticmethod
    def deconvolve_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, relu_activation

    @staticmethod
    def deconvolve_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        leaky_relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, leaky_relu_activation

    def forward(self, x):

        #print('Decoder: ', x.size())
        #x = self.relu(self.batch_norm_vae_input(self.linear_vae_input(x)))

        x = self.linear_1(x)
        # print('here')
        x = self.linear_1_batch_norm1D(x)
        x = self.relu(x)


        x = self.linear_2(x)
        # print('here')
        x = self.linear_2_batch_norm1D(x)
        x = self.relu(x)

        x = self.reshape(x)

        #print('Before Deconvolution: ', x.size())
        x = self.deconv2D_1(x)
        x = self.batch_norm2D_1(x)
        #print('After deconvolving:', x.size())
        x = self.relu_1(x)

        x = self.deconv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        #print('After deconvolving:', x.size())
        x = self.batch_norm2D_2(x)
        x = self.relu_2(x)

        x = self.deconv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        #print('After deconvolving:', x.size())
        x = self.batch_norm2D_3(x)
        x = self.relu_3(x)

        x = self.deconv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        x = self.batch_norm2D_4(x)
        #print('After deconvolving:', x.size())
        x = self.sigmoid(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear')

        return x

class DecoderEfficientNet(nn.Module):
    """
    Simple convolution decoder implementation
    """

    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=768, out_features=1024)
        self.batch_norm1D_1 = nn.BatchNorm1d(num_features=1024, momentum=0.01)

        self.linear_2 = nn.Linear(in_features=1024, out_features=25088)
        self.batch_norm1D_2 = nn.BatchNorm1d(num_features=25088, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.reshape = Reshape(-1, 128, 14, 14)

        self.deconv2D_1, self.batch_norm2D_1, self.relu_1 = self.deconvolve_batch_norm_unit(
            in_channels_=128,
            out_channels_=64,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_2, self.batch_norm2D_2, self.relu_2 = self.deconvolve_batch_norm_unit(
            in_channels_=64,
            out_channels_=32,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_3, self.batch_norm2D_3, self.relu_3 = self.deconvolve_batch_norm_unit(
            in_channels_=32,
            out_channels_=16,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

        self.deconv2D_4, self.batch_norm2D_4, self.relu_4 = self.deconvolve_batch_norm_unit(
            in_channels_=16,
            out_channels_=3,
            deconv_kernel_size=(2, 2),
            stride_=(2, 2),
        )

    @staticmethod
    def deconvolve_batch_norm_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        batch_norm_layer = nn.BatchNorm2d(
            num_features=out_channels_, momentum=0.01)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, batch_norm_layer, relu_activation

    @staticmethod
    def deconvolve_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)
        relu_activation = nn.ReLU(inplace=True)

        return deconvolution_layer, relu_activation

    @staticmethod
    def deconvolve_leaky_relu_unit(in_channels_, out_channels_, deconv_kernel_size, stride_):
        deconvolution_layer = nn.ConvTranspose2d(
            in_channels=in_channels_, out_channels=out_channels_, kernel_size=deconv_kernel_size, stride=stride_)

        leaky_relu_activation = nn.LeakyReLU(inplace=True)

        return deconvolution_layer, leaky_relu_activation

    def forward(self, x):

        #print('Decoder: ', x.size())
        #x = self.relu(self.batch_norm_vae_input(self.linear_vae_input(x)))

        x = self.linear_1(x)
        # print('here')
        x = self.batch_norm1D_1(x)
        x = self.relu(x)


        x = self.linear_2(x)
        # print('here')
        x = self.batch_norm1D_2(x)
        x = self.relu(x)

        x = self.reshape(x)

        #print('Before Deconvolution: ', x.size())
        x = self.deconv2D_1(x)
        x = self.batch_norm2D_1(x)
        #print('After deconvolving:', x.size())
        x = self.relu_1(x)

        x = self.deconv2D_2(x)  # 16, 222, 222 ---> 32, 222, 222
        #print('After deconvolving:', x.size())
        x = self.batch_norm2D_2(x)
        x = self.relu_2(x)

        x = self.deconv2D_3(x)  # 32, 219, 219 ---> 64, 220, 220
        #print('After deconvolving:', x.size())
        x = self.batch_norm2D_3(x)
        x = self.relu_3(x)

        x = self.deconv2D_4(x)  # 64, 110, 110 ---> 128, 108, 108
        x = self.batch_norm2D_4(x)
        #print('After deconvolving:', x.size())
        x = self.sigmoid(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear')

        return x
