import functools

import torch
from torch import nn

from models.modules.mobile_modules import SeparableConv2d

#from models.networks import WBlock, NBlock
from ...networks import init_net

import math
import sys

class MobileResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(MobileResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
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

        conv_block += [
            SeparableConv2d(in_channels=dim, out_channels=dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(dim), nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=dim, out_channels=dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MobileResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9, padding_type='reflect', decoder=True,
                 wplus=True, init_type='normal', init_gain=0.02, gpu_ids=[],
                 img_size=128, img_size_dec=128):
        assert (n_blocks >= 0)
        self.decoder = decoder
        self.wplus = wplus
        super(MobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            model += [MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate,
                                        use_bias=use_bias)]

        for i in range(n_blocks2):
            model += [MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate,
                                        use_bias=use_bias)]

        for i in range(n_blocks3):
            model += [MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate,
                                        use_bias=use_bias)]

        if self.decoder:
            for i in range(n_downsampling):  # add upsampling layers
                mult = 2 ** (n_downsampling - i)
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
                model += [nn.ReflectionPad2d(3)]
                model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
                model += [nn.Tanh()]
                self.model = nn.Sequential(*model)
        else:
            if wplus == False:
                n_feat = 4096 #1024 # 256
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

                
    def forward(self, input):
        """Standard forward"""
        # input = input.clamp(-1, 1)
        # for i, module in enumerate(self.model):
        #     print(i, input.size())
        #     print(module)
        #     if isinstance(module, nn.Conv2d):
        #         print(module.stride)
        #     input = module(input)
        # return input
        if self.decoder:
            return self.model(input)
        else:
            output = self.model(input)
            if not self.wplus:
                output = self.conv(output).squeeze(dim=1)
                output = torch.flatten(output,1)
                output = self.to_w(output).unsqueeze(dim=0)
                return output
            else:
                outputs = []
                noutputs = []
                for wc in self.wblocks:
                    outputs.append(wc(output))
                for nc in self.nblocks:
                    noutputs.append(nc(output))
                return outputs, noutputs

class WBlock(nn.Module):
    """Define a linear block for W"""
    def __init__(self, dim, n_feat, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super(WBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=dim,out_channels=1,kernel_size=1)
        self.lin1 = nn.Linear(n_feat,32,bias=True)
        self.lin2 = nn.Linear(32,512,bias=True)
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
            self.n_block = [SeparableConv2d(in_channels=256,out_channels=32,kernel_size=3,stride=1,padding=1),
                            nn.InstanceNorm2d(1),
                            nn.ReLU(True)]
            self.n_block += [nn.Upsample((out_feat,out_feat))]
            self.n_block += [nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)]
            self.n_block += [nn.Flatten()]
            self.n_block = init_net(nn.Sequential(*self.n_block), init_type, init_gain, gpu_ids)
                    
    def forward(self, x):
        out = self.n_block(x)
        return torch.reshape(out.unsqueeze(1),(1,1,self.out_feat,self.out_feat))
