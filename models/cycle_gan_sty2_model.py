import sys
import torch
import itertools
from util.image_pool import ImagePool
from util.losses import L1_Charbonnier_loss
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
from models.vgg_perceptual_loss import VGGPerceptualLoss

import random
import math
from torch import distributed as dist

class CycleGANSty2Model(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_G', type=float, default=1.0, help='weight for generator loss')
            parser.add_argument('--D_noise', action='store_true', help='whether to add instance noise to discriminator inputs')
            parser.add_argument('--D_label_smooth', action='store_true', help='whether to use one-sided label smoothing with discriminator')
            parser.add_argument('--rec_noise', action='store_true', help='whether to add noise to reconstruction')
            parser.add_argument('--wplus', action='store_true', help='whether to work in W+ latent space')
            parser.add_argument('--wskip', action='store_true', help='whether to use skip connections to latent wplus heads')
            parser.add_argument('--truncation',type=float,default=1,help='whether to use truncation trick (< 1)')
            parser.add_argument('--decoder_size', type=int, default=512)
            parser.add_argument('--d_reg_every', type=int, default=16)
            parser.add_argument('--g_reg_every', type=int, default=4)
            parser.add_argument('--r1', type=float, default=10)
            #parser.add_argument('--mixing', type=float, default=0.9)
            parser.add_argument('--path_batch_shrink', type=int, default=2)
            parser.add_argument('--path_regularize', type=float, default=2)
            parser.add_argument('--no_init_weigth_D_sty2', action='store_true')
            parser.add_argument('--no_init_weigth_dec_sty2', action='store_true')
            parser.add_argument('--no_init_weigth_G', action='store_true')
            parser.add_argument('--load_weigth_decoder', action='store_true')
    
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        losses = ['G_A','G_B']
        losses += ['D_A', 'D_B']
        losses=[]
        losses += ['cycle_A', 'idt_A', 
                   'cycle_B', 'idt_B']
        losses += ['g_nonsaturating_A','g_nonsaturating_B']

        if self.opt.g_reg_every != 0:
            losses +=['weighted_path_A','weighted_path_B']

        losses+= ['d_dec_A','d_dec_B']

        if self.opt.d_reg_every != 0:
            losses += ['grad_pen_A','grad_pen_B']
                  
        self.loss_names = losses
        self.truncation = opt.truncation
        self.r1 = opt.r1
        
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
           visual_names_A.append('idt_B')
           visual_names_B.append('idt_A') # beniz: inverted for original

        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        print('define gen')
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids, decoder=False, wplus=opt.wplus, wskip=opt.wskip,img_size=self.opt.decoder_size)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids, decoder=False, wplus=opt.wplus, wskip=opt.wskip,img_size=self.opt.decoder_size)

        # Define stylegan2 decoder
        print('define decoder')
        self.netDecoderG_A = networks.define_decoder(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,size=self.opt.decoder_size,init_weight=not self.opt.no_init_weigth_dec_sty2)
        self.netDecoderG_B = networks.define_decoder(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,size=self.opt.decoder_size,init_weight=not self.opt.no_init_weigth_dec_sty2)
        
        # Load pretrained weights stylegan2 decoder
        
        nameDGA = 'DecoderG_A'
        nameDGB = 'DecoderG_B'
        if self.opt.load_weigth_decoder:
            load_filename = 'network_A.pt'
            load_path = os.path.join(self.save_dir, load_filename)
        
            net = getattr(self, 'net' + nameDGA)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict['g_ema'])
            self.set_requires_grad(net, True)
                                
            load_filename = 'network_B.pt'
            load_path = os.path.join(self.save_dir, load_filename)
        
            net = getattr(self, 'net' + nameDGB)
            
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict['g_ema'])
            self.set_requires_grad(net, True)

        if self.opt.truncation < 1:
            self.mean_latent_A = self.netDecoderG_A.module.mean_latent(4096)
            self.mean_latent_B = self.netDecoderG_B.module.mean_latent(4096)
        else:
            self.mean_latent_A = None
            self.mean_latent_B = None
                                        
        self.model_names += [nameDGA,nameDGB]
    
        print('define dis dec')
        self.netDiscriminatorDecoderG_A = networks.define_discriminatorstylegan2(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,init_weight=not self.opt.no_init_weigth_D_sty2,img_size=self.opt.crop_size)
        self.model_names += ['DiscriminatorDecoderG_A']

        self.netDiscriminatorDecoderG_B = networks.define_discriminatorstylegan2(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,init_weight=not self.opt.no_init_weigth_D_sty2,img_size=self.opt.crop_size)
        self.model_names += ['DiscriminatorDecoderG_B']
                
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.real_A_pool = ImagePool(opt.pool_size)
            self.real_B_pool = ImagePool(opt.pool_size)
                 
            # define loss functions
            if opt.D_label_smooth:
                target_real_label = 0.9
            else:
                target_real_label = 1.0
            self.criterionGAN = networks.GANLoss(opt.gan_mode,target_real_label=target_real_label).to(self.device)
            #self.criterionCycle = torch.nn.L1Loss()
            #self.criterionIdt = torch.nn.L1Loss()
            self.criterionCycle = VGGPerceptualLoss().cuda()
            self.criterionCycle2 = torch.nn.MSELoss()
            self.criterionIdt = VGGPerceptualLoss().cuda()
            self.criterionIdt2 = torch.nn.MSELoss()
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),self.netDecoderG_A.parameters(), self.netDecoderG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Decoder = torch.optim.Adam(itertools.chain(self.netDiscriminatorDecoderG_A.parameters(),self.netDiscriminatorDecoderG_B.parameters()),
                                            lr=opt.D_lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)

            self.rec_noise = opt.rec_noise
            self.stddev = 0.1
            self.D_noise = opt.D_noise

            self.niter=0
            self.mean_path_length_A = 0
            self.mean_path_length_B = 0
            
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.z_fake_B = self.netG_A(self.real_A)
        d = 1
        #self.netDecoderG_A.eval()
        self.fake_B,self.latent_fake_B = self.netDecoderG_A(self.z_fake_B,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_A,randomize_noise=False,return_latents=True)
        
        if self.isTrain:
            #self.netDecoderG_B.eval()
            if self.rec_noise:
                self.fake_B_noisy1 = self.gaussian(self.fake_B)
                self.z_rec_A= self.netG_B(self.fake_B_noisy1)
            else:
                self.z_rec_A = self.netG_B(self.fake_B)
            self.rec_A = self.netDecoderG_B(self.z_rec_A,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_B, randomize_noise=False)[0]
                
            self.z_fake_A = self.netG_B(self.real_B)
            self.fake_A,self.latent_fake_A = self.netDecoderG_B(self.z_fake_A,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_B,randomize_noise=False,return_latents=True)
            
            if self.rec_noise:
                self.fake_A_noisy1 = self.gaussian(self.fake_A)
                self.z_rec_B = self.netG_A(self.fake_A_noisy1)
            else:
                self.z_rec_B = self.netG_A(self.fake_A)
            self.rec_B = self.netDecoderG_A(self.z_rec_B,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_A, randomize_noise=False)[0]
            
    def backward_G(self):
        #print('BACKWARD G')
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_G = self.opt.lambda_G
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.z_idt_A = self.netG_A(self.real_B)
            self.idt_A = self.netDecoderG_A(self.z_idt_A,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_A,randomize_noise=False)[0]
            
            self.loss_idt_A = (self.criterionIdt(self.idt_A, self.real_B)
                               + self.criterionIdt2(self.idt_A, self.real_B)) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.z_idt_B = self.netG_B(self.real_A)
            self.idt_B = self.netDecoderG_B(self.z_idt_B,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_B,randomize_noise=False)[0]
            self.loss_idt_B = (self.criterionIdt(self.idt_B, self.real_A)
                               + self.criterionIdt2(self.idt_B, self.real_A)) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Forward cycle loss
        self.loss_cycle_A = (self.criterionCycle(self.rec_A, self.real_A) + self.criterionCycle2(self.rec_A, self.real_A)) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = (self.criterionCycle(self.rec_B, self.real_B) + self.criterionCycle2(self.rec_B, self.real_B)) * lambda_B
        # combined loss standard cyclegan
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        compute_g_regularize = True
        if self.opt.path_regularize == 0.0 or self.opt.g_reg_every == 0 or not self.niter % self.opt.g_reg_every == 0 :
            #self.loss_weighted_path_A = 0* self.loss_weighted_path_A
            #self.loss_weighted_path_B = 0* self.loss_weighted_path_B
            compute_g_regularize = False
        
        #A
        self.fake_pred_g_loss_A = self.netDiscriminatorDecoderG_A(self.fake_A)
        self.loss_g_nonsaturating_A = self.g_nonsaturating_loss(self.fake_pred_g_loss_A)
        
        if compute_g_regularize:
            self.path_loss_A, self.mean_path_length_A, self.path_lengths_A = self.g_path_regularize(
                self.fake_A, self.latent_fake_A, self.mean_path_length_A
            )

            self.loss_weighted_path_A = self.opt.path_regularize * self.opt.g_reg_every * self.path_loss_A
        
            if self.opt.path_batch_shrink:
                self.loss_weighted_path_A += 0 * self.fake_A[0, 0, 0, 0]

            self.mean_path_length_avg_A = (
                self.reduce_sum(self.mean_path_length_A).item() / self.get_world_size()
            )
        else:
            self.loss_weighted_path_A = 0#*self.loss_weighted_path_A

        #B
        self.fake_pred_g_loss_B = self.netDiscriminatorDecoderG_B(self.fake_B)
        self.loss_g_nonsaturating_B = self.g_nonsaturating_loss(self.fake_pred_g_loss_B)
        
        if compute_g_regularize:
            self.path_loss_B, self.mean_path_length_B, self.path_lengths_B = self.g_path_regularize(
                self.fake_B, self.latent_fake_B, self.mean_path_length_B
            )

            self.loss_weighted_path_B = self.opt.path_regularize * self.opt.g_reg_every * self.path_loss_B
        
            if self.opt.path_batch_shrink:
                #self.loss_weighted_path_B += 0 * self.fake_img_path_loss_B[0, 0, 0, 0]
                self.loss_weighted_path_B += 0 * self.fake_B[0, 0, 0, 0]

            self.mean_path_length_avg_B = (
                self.reduce_sum(self.mean_path_length_B).item() / self.get_world_size()
            )
        else:
            self.loss_weighted_path_B = 0#*self.loss_weighted_path_B

        self.loss_G += self.opt.lambda_G*(self.loss_g_nonsaturating_A + self.loss_g_nonsaturating_B)

        if not self.opt.path_regularize == 0.0 and not self.opt.g_reg_every == 0 and self.niter % self.opt.g_reg_every == 0 :
            self.loss_G += self.loss_weighted_path_A + self.loss_weighted_path_B
        
        self.loss_G.backward()

    def backward_discriminator_decoder(self):
        real_pred_A = self.netDiscriminatorDecoderG_A(self.real_A)
        fake_pred_A = self.netDiscriminatorDecoderG_A(self.fake_A_pool.query(self.fake_A))
        self.loss_d_dec_A = self.d_logistic_loss(real_pred_A,fake_pred_A).unsqueeze(0)

        real_pred_B = self.netDiscriminatorDecoderG_B(self.real_B)
        fake_pred_B = self.netDiscriminatorDecoderG_B(self.fake_B_pool.query(self.fake_B))
        self.loss_d_dec_B = self.d_logistic_loss(real_pred_B,fake_pred_B).unsqueeze(0)
        self.loss_d_dec = self.loss_d_dec_A + self.loss_d_dec_B
        
        if self.opt.d_reg_every != 0:
            if self.niter %self.opt.d_reg_every == 0:
                temp = real_pred_A/real_pred_A.detach()
        
                cur_real_A = self.real_A_pool.query(self.real_A)
                cur_real_A.requires_grad = True
                real_pred_A_2 = self.netDiscriminatorDecoderG_A(cur_real_A)

                self.loss_grad_pen_A = self.gradient_penalty(cur_real_A,real_pred_A_2,self.r1)
                
                cur_real_B = self.real_B_pool.query(self.real_B)
                cur_real_B.requires_grad = True
                real_pred_B_2 = self.netDiscriminatorDecoderG_B(cur_real_B)
            
                self.loss_grad_pen_B = self.gradient_penalty(cur_real_B,real_pred_B_2,self.r1)
            else:
                self.loss_grad_pen_A = 0
                self.loss_grad_pen_B = 0

            self.loss_d_dec += self.loss_grad_pen_A + self.loss_grad_pen_B

        self.loss_d_dec.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netDiscriminatorDecoderG_A,self.netDiscriminatorDecoderG_B], False)
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.set_requires_grad([self.netDecoderG_A, self.netDecoderG_B], True)
        self.netDecoderG_A.zero_grad()
        self.netDecoderG_B.zero_grad()
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.optimizer_D_Decoder.zero_grad()
        self.niter = self.niter +1
        self.set_requires_grad([self.netDiscriminatorDecoderG_A,self.netDiscriminatorDecoderG_B], True)
        self.backward_discriminator_decoder()
        self.optimizer_D_Decoder.step()
        #self.set_requires_grad([self.netDiscriminatorDecoderG_A,self.netDiscriminatorDecoderG_B], False)

    def gaussian(self, in_tensor):
        noisy_image = torch.zeros(list(in_tensor.size())).data.normal_(0, self.stddev).cuda() + in_tensor
        # noisy_tensor = 2 * (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min()) - 1
        return noisy_image


    def d_logistic_loss(self,real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()


    def d_r1_loss(self,real_pred, real_img):
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img#, create_graph=True,allow_unused=True
        )
        
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        
        return grad_penalty


    def g_nonsaturating_loss(self,fake_pred):
        loss = F.softplus(-fake_pred).mean()
        return loss


    def g_path_regularize(self,fake_img, latents, mean_path_length, decay=0.01):
        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3]
        )

        noise.requires_grad=True
        
        grad, = torch.autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True#,allow_unused=True
        )
        #print(grad)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_mean.detach(), path_lengths

    def make_noise(self,batch, latent_dim, n_noise, device):
        
        if n_noise == 1:
            return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device)#.unbind(0)

        return noises

    def mixing_noise(self,batch, latent_dim, prob, device):
        log_size = int(math.log(128, 2))
        n_latent = log_size * 2 - 2
        temp = random.random()
        temp_noise = self.make_noise(batch, latent_dim, 2, device)
        if prob > 0 and temp < prob:
            inject_index = random.randint(1, n_latent - 1)
        else:
            inject_index = n_latent
        latent = temp_noise[0].unsqueeze(1).repeat(1, inject_index, 1)
        latent2 = temp_noise[1].unsqueeze(1).repeat(1, n_latent - inject_index, 1)
        latent = torch.cat([latent, latent2], 1)
        latents = []
        return latent

    def reduce_sum(self,tensor):
        if not dist.is_available():
            return tensor

        if not dist.is_initialized():
            return tensor

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        return tensor

    def get_world_size(self):
        if not dist.is_available():
            return 1

        if not dist.is_initialized():
            return 1

        return dist.get_world_size()

    def gradient_penalty(self,images, output, weight = 10):
        batch_size = images.shape[0]
        gradients = torch.autograd.grad(outputs=output, inputs=images,
                               grad_outputs=torch.ones(output.size()).cuda(),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

