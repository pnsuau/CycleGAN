import sys
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision
from torch.utils import model_zoo
from torchvision.models import vgg
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .UNet import UNet
from .decoder_stylegan2 import Generator as GeneratorStyleGAN2, Discriminator as DiscriminatorStyleGAN2, EqualLinear as EqualLinear
import math
###############################################################################
# Helper Functions
###############################################################################


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[],init_weight=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_weight:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[], decoder=True, wplus=True, wskip=False, init_weight=True, img_size=128, img_size_dec=128,nb_attn = 10,nb_mask_input=2):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        use_spectral (bool) -- if use spectral norm.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral, n_blocks=9, decoder=decoder, wplus=wplus, wskip=wskip, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, img_size=img_size,img_size_dec=img_size_dec)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral, n_blocks=6, decoder=decoder, wplus=wplus, wskip=wskip, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, img_size=img_size,img_size_dec=img_size_dec)
    elif netG == 'resnet_12blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral, n_blocks=12, decoder=decoder, wplus=wplus, wskip=wskip, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, img_size=img_size,img_size_dec=img_size_dec)
    elif netG == 'mobile_resnet_9blocks':
        from .modules.resnet_architecture.mobile_resnet_generator import MobileResnetGenerator
        net = MobileResnetGenerator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer,
                                    dropout_rate=0, n_blocks=9, decoder=decoder, wplus=wplus,
                                    init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids,
                                    img_size=img_size, img_size_dec=img_size_dec)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'resnet_attn':
        net = ResnetGenerator_attn(input_nc, output_nc, ngf, n_blocks=9, use_spectral=use_spectral)
    elif netG == 'resnet_attn_jb':
        net = ResnetGenerator_attn2(input_nc, output_nc, ngf, n_blocks=9, use_spectral=use_spectral,nb_attn = nb_attn,nb_mask_input=nb_mask_input)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    #if len(gpu_ids) > 0:
        #assert(torch.cuda.is_available())
        #net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    print('init type before',init_type)
    return init_net(net, init_type, init_gain, gpu_ids,init_weight=init_weight)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_dropout=False, use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        use_dropout (bool) -- whether to use dropout layers
        use_spectral(bool) -- whether to use spectral norm
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_C(output_nc, ndf, init_type='normal', init_gain=0.02, gpu_ids=[], nclasses=10):
    print('nclasses=',nclasses)
    netC = Classifier(output_nc, ndf, nclasses)
    return init_net(netC, init_type, init_gain, gpu_ids)

def define_f(input_nc, nclasses, init_type='normal', init_gain=0.02, gpu_ids=[], fs_light=False):
    if not fs_light:
        net = VGG16_FCN8s(nclasses,pretrained = False, weights_init =None,output_last_ft=False)
    else:
        net = UNet(classes=nclasses)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_discriminator(input_dim=4096, output_dim=2, pretrained=False, weights_init='', init_type='normal', init_gain=0.02, gpu_ids=[],init_weight=True):
    net = Discriminator(input_dim=4096, output_dim=2, pretrained=False, weights_init='')
    return init_net(net, init_type, init_gain, gpu_ids,init_weight=init_weight)

def define_classifier_w(pretrained=False, weights_init='', init_type='normal', init_gain=0.02, gpu_ids=[],init_weight=True,img_size_dec=256):
    net = Classifier_w(img_size_dec=img_size_dec)
    return init_net(net, init_type, init_gain, gpu_ids,init_weight=init_weight)

def define_decoder(init_type='normal', init_gain=0.02, gpu_ids=[],decoder=False,size=512,init_weight=True,clamp=False):
    net = GeneratorStyleGAN2(size,512,8,clamp=clamp)
    #if len(gpu_ids) > 0:
        #assert(torch.cuda.is_available())
        #net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    
    return init_net(net, init_type, init_gain, gpu_ids,init_weight=init_weight)


def define_discriminatorstylegan2(init_type='normal', init_gain=0.02, gpu_ids=[], init_weight=True, img_size=128, lightness=1):
    net = DiscriminatorStyleGAN2(img_size, lightness=lightness)
    #if len(gpu_ids) > 0:
        #assert(torch.cuda.is_available())
        #net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    
    return init_net(net, init_type, init_gain, gpu_ids,init_weight=init_weight)    

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_spectral=False, decoder=True, wplus=True, wskip=False, init_type='normal', init_gain=0.02, gpu_ids=[],img_size=128,img_size_dec=128):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.bpoints = []
        self.head = []
        self.decoder = decoder
        self.wplus = wplus
        self.wskip = wskip
        head = []
        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        fl = [nn.ReflectionPad2d(3),
                 spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),use_spectral),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        head += fl
        model += fl
        
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            dsp = [spectral_norm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),use_spectral),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            model += dsp
            head += dsp

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            resblockl = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            model += resblockl
            self.bpoints += resblockl

        if self.decoder:
            for i in range(n_downsampling):  # add upsampling layers
                mult = 2 ** (n_downsampling - i)
                model += [spectral_norm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                           kernel_size=3, stride=2,
                                                           padding=1, output_padding=1,
                                                           bias=use_bias),use_spectral),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
                model += [nn.ReflectionPad2d(3)]
                model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
                model += [nn.Tanh()]
            self.model = nn.Sequential(*model)
        else:
            if wplus == False:
                n_feat = 1024 # 256
                to_w = [nn.Linear(n_feat,img_size_dec)] # sty2 image output size
                self.to_w = nn.Sequential(*to_w)
                self.conv = nn.Conv2d(ngf*mult,1, kernel_size=1)
            else:
                n_feat = 2**(2*int(math.log(img_size,2)-2))
                self.n_wplus = (2*int(math.log(img_size_dec,2)-1))
                self.wblocks = nn.ModuleList()
                for n in range(0,self.n_wplus):
                    self.wblocks += [WBlock(ngf*mult,n_feat,init_type,init_gain,gpu_ids)]
                self.nblocks = nn.ModuleList()
                noise_map = [4,8,8,16,16,32,32,64,64,128,128,256,256,512,512,1024,1024]
                for n in range(0,self.n_wplus-1):
                    self.nblocks += [NBlock(ngf*mult,n_feat,noise_map[n],init_type,init_gain,gpu_ids)]
            self.model = nn.Sequential(*model)
            self.head = nn.Sequential(*head)
        
    def forward(self, input):
        """Standard forward"""
        if self.decoder:
            output = self.model(input)
        else:
            res_outputs = []
            output = self.head(input)
            for bp in self.bpoints:
                output = (bp(output))
                res_outputs.append(output)
            
        if hasattr(self,'to_w'):
            output = self.conv(output).squeeze(dim=1)
            output = torch.flatten(output,1)
            output = self.to_w(output).unsqueeze(dim=0)
            return output
        elif hasattr(self,'wblocks'):
            outputs = []
            noutputs = []
            if not self.wskip:
                for wc in self.wblocks:
                    outputs.append(wc(output))
                for nc in self.nblocks:
                    noutputs.append(nc(output))
            else:
                nou = 0
                for o in res_outputs: # skip connections to latent heads
                    outputs.append(self.wblocks[nou](o))
                    noutputs.append(self.nblocks[nou](o))
                    nou += 1
                for k in range(0,self.n_wplus-nou): # remaining latent heads
                    outputs.append(self.wblocks[nou+k](output))
                for k in range(0,self.n_wplus-nou-1):
                    noutputs.append(self.nblocks[nou+k](output))
            return outputs, noutputs
        else:
            return output

class ResnetGenerator_attn(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, use_spectral=False):
        super(ResnetGenerator_attn, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.conv1 = spectral_norm(nn.Conv2d(input_nc, ngf, 7, 1, 0),use_spectral)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = spectral_norm(nn.Conv2d(ngf, ngf * 2, 3, 2, 1),use_spectral)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),use_spectral)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block_attn(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),use_spectral)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),use_spectral)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)
        self.deconv3_content = spectral_norm(nn.Conv2d(ngf, 27, 7, 1, 0),use_spectral)

        self.deconv1_attention = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),use_spectral)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),use_spectral)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf, 10, 1, 1, 0)
        
        self.tanh = torch.nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(x_content)
        image = self.tanh(content)
        image1 = image[:, 0:3, :, :]
        # print(image1.size()) # [1, 3, 256, 256]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        # image10 = image[:, 27:30, :, :]

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        # x_attention = F.pad(x_attention, (3, 3, 3, 3), 'reflect')
        # print(x_attention.size()) [1, 64, 256, 256]
        attention = self.deconv3_attention(x_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        attention1 = attention1_.repeat(1, 3, 1, 1)
        # print(attention1.size())
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        # output10 = image10 * attention10
        output10 = input * attention10

        o=output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10

        return o#, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, attention1,attention2,attention3, attention4, attention5, attention6, attention7, attention8,attention9,attention10, image1, image2,image3,image4,image5,image6,image7,image8,image9

class ResnetGenerator_attn2(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[],size=128,nb_attn = 10,nb_mask_input=1): #nb_attn : nombre de masques d'attention, nb_mask_input : nb de masques d'attention qui vont etre appliqués a l'input
        super(ResnetGenerator_attn2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.nb_attn = nb_attn
        self.nb_mask_input = nb_mask_input
        self.conv1 = spectral_norm(nn.Conv2d(input_nc, ngf, 7, 1, 0),use_spectral)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = spectral_norm(nn.Conv2d(ngf, ngf * 2, 3, 2, 1),use_spectral)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),use_spectral)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block_attn(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),use_spectral)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),use_spectral)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)        
        self.deconv3_content = spectral_norm(nn.Conv2d(ngf, 3 * (self.nb_attn-nb_mask_input), 7, 1, 0),use_spectral)#self.nb_attn-nb_mask_input: nombre d'images générées ou les masques d'attention vont etre appliqués

        self.deconv1_attention = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),use_spectral)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),use_spectral)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf,self.nb_attn, 1, 1, 0)
        
        self.tanh = torch.nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(x_content)
        image = self.tanh(content)

        images = []

        #for i in range(self.n_wplus - self.nb_mask_input):
        for i in range(0,9):
            images.append(image[:, 3*i:3*(i+1), :, :])

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        # x_attention = F.pad(x_attention, (3, 3, 3, 3), 'reflect')
        # print(x_attention.size()) [1, 64, 256, 256]
        attention = self.deconv3_attention(x_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        attentions =[]
        
        #for i in range(self.n_wplus):
        for i in range(self.nb_attn):
            attentions.append(attention[:, i:i+1, :, :].repeat(1, 3, 1, 1))

        outputs = []

        #for i in range(self.n_wplus):
        #    if i < self.nb_mask_input:
        #        outputs.append(input*attentions[i])
        #    else:
        #        outputs.append(images[i-self.nb_mask_input]*attentions[i])
        #output1 = input * attention1
        
        for i in range(self.nb_attn-self.nb_mask_input):
            outputs.append(images[i]*attentions[i])
        for i in range(self.nb_attn-self.nb_mask_input,self.nb_attn):
            outputs.append(input * attentions[i])
        
        o = outputs[0]
        for i in range(1,self.nb_attn):
            o += outputs[i]
        return o
            
    
class resnet_block_attn(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block_attn, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class WBlock(nn.Module):
    """Define a linear block for W"""
    def __init__(self, dim, n_feat, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super(WBlock, self).__init__()
        self.conv2d = nn.Conv2d(dim,1,kernel_size=1)
        self.lin1 = nn.Linear(n_feat,32,bias=True)
        self.lin2 = nn.Linear(32,512,bias=True)
        #self.lin = nn.Linear(n_feat,512)
        w_block = []
        w_block += [self.conv2d,nn.InstanceNorm2d(1),nn.Flatten(),self.lin1,nn.ReLU(True),self.lin2]
        self.w_block = init_net(nn.Sequential(*w_block), init_type, init_gain, gpu_ids)
        
    def forward(self, x):
        out = self.w_block(x)
        return out
    
class NBlock(nn.Module):
    """Define a linear block for N"""
    def __init__(self, dim, n_feat, out_feat, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super(NBlock, self).__init__()
        self.out_feat = out_feat
        if out_feat < 32: # size of input
            self.conv2d = nn.Conv2d(dim,1,kernel_size=1)
            self.lin = nn.Linear(n_feat,out_feat**2)
            n_block = []
            n_block += [self.conv2d,nn.InstanceNorm2d(1),nn.Flatten(),self.lin]
            self.n_block = init_net(nn.Sequential(*n_block), init_type, init_gain, gpu_ids)
        else:
            self.n_block = []
            self.n_block = [nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1),
                            nn.InstanceNorm2d(1),
                            nn.ReLU(True)]
            self.n_block += [nn.Upsample((out_feat,out_feat))]
            self.n_block += [nn.Conv2d(64,1,kernel_size=1)]
            self.n_block += [nn.Flatten()]
            self.n_block = init_net(nn.Sequential(*self.n_block), init_type, init_gain, gpu_ids)
                    
    def forward(self, x):
        out = self.n_block(x)
        return torch.reshape(out.unsqueeze(1),(1,1,self.out_feat,self.out_feat))
        
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral=False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),use_spectral), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),use_spectral), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Classifier_w(nn.Module):                                                                             
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[],img_size_dec=256):
        super(Classifier_w, self).__init__()
        n_w_plus = 2*int(math.log(img_size_dec,2)-1)
        model = [nn.Flatten(),nn.utils.spectral_norm(nn.Linear(n_w_plus*512,1)),nn.LeakyReLU(0.2,True)]
        self.model = init_net(nn.Sequential(*model), init_type, init_gain, gpu_ids)
        
    def forward(self, x):
        out = self.model(x.permute(1,0,2))
        return out
    
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False, use_spectral=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            use_dropout (bool) -- whether to use dropout layers
            use_spectral (bool) -- whether to use spectral norm
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),use_spectral), nn.LeakyReLU(0.2, True)]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),use_spectral),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            if use_dropout:
                sequence += [nn.Dropout(0.5)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),use_spectral),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),use_spectral)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class Classifier(nn.Module):
    def __init__(self, input_nc, ndf, nclasses, norm_layer=nn.BatchNorm2d):
        super(Classifier, self).__init__()

        kw = 3
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2),
                norm_layer(ndf * nf_mult, affine=True), #beniz: pb with dimensions with batch_size = 1
                nn.LeakyReLU(0.2, True)
            ]
        self.before_linear = nn.Sequential(*sequence)
        
        sequence = [
            nn.Linear(ndf * nf_mult, 1024),
            nn.Linear(1024, nclasses)
        ]

        self.after_linear = nn.Sequential(*sequence)
    
    def forward(self, x):
        bs = x.size(0)
        #print(x)
        out = self.after_linear(self.before_linear(x).view(bs, -1))
        return out

###################### CYCADA ###################################

    
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class Bilinear(nn.Module):

    def __init__(self, factor, num_channels):
        super().__init__()
        self.factor = factor
        filter = get_upsample_filter(factor * 2)
        w = torch.zeros(num_channels, num_channels, factor * 2, factor * 2)
        for i in range(num_channels):
            w[i, i] = filter
        self.register_buffer('w', w)

    def forward(self, x):
        return F.conv_transpose2d(x, Variable(self.w), stride=self.factor)


class VGG16_FCN8s(nn.Module):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, num_cls=19, pretrained=True, weights_init=None, 
            output_last_ft=False):
        super().__init__()
        self.output_last_ft = output_last_ft
        self.vgg = make_layers(vgg.cfgs['D'])
        self.vgg_head = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, num_cls, 1)
            )
        self.upscore2 = self.upscore_pool4 = Bilinear(2, num_cls)
        self.upscore8 = Bilinear(8, num_cls)
        self.score_pool4 = nn.Conv2d(512, num_cls, 1)
        for param in self.score_pool4.parameters():
            init.constant_(param, 0)
        self.score_pool3 = nn.Conv2d(256, num_cls, 1)
        for param in self.score_pool3.parameters():
            init.constant_(param, 0)
        
        if pretrained:
            if weights_init is not None:
                self.load_weights(torch.load(weights_init))
            else:
                self.load_base_weights()
 
    def load_base_vgg(self, weights_state_dict):
        vgg_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg.')
        self.vgg.load_state_dict(vgg_state_dict)
     
    def load_vgg_head(self, weights_state_dict):
        vgg_head_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg_head.') 
        self.vgg_head.load_state_dict(vgg_head_state_dict)
    
    def get_dict_by_prefix(self, weights_state_dict, prefix):
        return {k[len(prefix):]: v 
                for k,v in weights_state_dict.items()
                if k.startswith(prefix)}


    def load_weights(self, weights_state_dict):
        self.load_base_vgg(weights_state_dict)
        self.load_vgg_head(weights_state_dict)

    def split_vgg_head(self):
        self.classifier = list(self.vgg_head.children())[-1]
        self.vgg_head_feat = nn.Sequential(*list(self.vgg_head.children())[:-1])


    def forward(self, x):
        input = x
        x = F.pad(x, (99, 99, 99, 99), mode='constant', value=0)
        intermediates = {}
        fts_to_save = {16: 'pool3', 23: 'pool4'}
        for i, module in enumerate(self.vgg):
            x = module(x)
            if i in fts_to_save:
                intermediates[fts_to_save[i]] = x
       
        ft_to_save = 5 # Dropout before classifier
        last_ft = {}
        for i, module in enumerate(self.vgg_head):
            x = module(x)
            if i == ft_to_save:
                last_ft = x      
        
        _, _, h, w = x.size()
        upscore2 = self.upscore2(x)
        pool4 = intermediates['pool4']
        score_pool4 = self.score_pool4(0.01 * pool4)
        score_pool4c = _crop(score_pool4, upscore2, offset=5)
        fuse_pool4 = upscore2 + score_pool4c
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        pool3 = intermediates['pool3']
        score_pool3 = self.score_pool3(0.0001 * pool3)
        score_pool3c = _crop(score_pool3, upscore_pool4, offset=9)
        fuse_pool3 = upscore_pool4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)
        score = _crop(upscore8, input, offset=31)
        if self.output_last_ft: 
            return score, last_ft
        else:
            return score


    def load_base_weights(self):
        """This is complicated because we converted the base model to be fully
        convolutional, so some surgery needs to happen here."""
        base_state_dict = model_zoo.load_url(vgg.model_urls['vgg16'])
        vgg_state_dict = {k[len('features.'):]: v
                          for k, v in base_state_dict.items()
                          if k.startswith('features.')}
        self.vgg.load_state_dict(vgg_state_dict)
        vgg_head_params = self.vgg_head.parameters()
        for k, v in base_state_dict.items():
            if not k.startswith('classifier.'):
                continue
            if k.startswith('classifier.6.'):
                # skip final classifier output
                continue
            vgg_head_param = next(vgg_head_params)
            vgg_head_param.data = v.view(vgg_head_param.size())

class Discriminator(nn.Module):
    def __init__(self, input_dim=4096, output_dim=2, pretrained=False, weights_init=''):
        super().__init__()
        dim1 = 1024 if input_dim==4096 else 512
        dim2 = int(dim1/2)
        self.D = nn.Sequential(
            nn.Conv2d(input_dim, dim1, 1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, 1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim2, output_dim, 1)
            )

        if pretrained and weights_init is not None:
            self.load_weights(weights_init)

    def forward(self, x):
        d_score = self.D(x)
        return d_score

    def load_weights(self, weights):
        print('Loading discriminator weights')
        self.load_state_dict(torch.load(weights))


class Transform_Module(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(input_dim, input_dim, 1),                                                                                                                                                            
            #nn.ReLU(inplace=True),                                                                                                                                                                         
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_eye(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        t_x = self.transform(x)
        return t_x


def init_eye(tensor):
    if isinstance(tensor, Variable):
        init_eye(tensor.data)
        return tensor
    return tensor.copy_(torch.eye(tensor.size(0), tensor.size(1)))


def _crop(input, shape, offset=0):
    _, _, h, w = shape.size()
    return input[:, :, offset:offset + h, offset:offset + w].contiguous()


def make_layers(cfg, batch_norm=False):
    """This is almost verbatim from torchvision.models.vgg, except that the                                                                                                                                 
    MaxPool2d modules are configured with ceil_mode=True.                                                                                                                                                   
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            modules = [conv2d, nn.ReLU(inplace=True)]
            if batch_norm:
                modules.insert(1, nn.BatchNorm2d(v))
            layers.extend(modules)
            in_channels = v
    return nn.Sequential(*layers)
