import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util
from .ref_losses import Styleloss_vgg_mean_variance, ContentLoss_constra, ContentLoss_vgg, StyleLoss_constra, StyleLoss_constra_rhr3, StyleLoss_constra_1
import itertools
from . import vgg19
import torch.nn as nn
from util.image_pool import ImagePool
from .weight import sigmoid_rampup
# import pyiqa



class REFCYCLEModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # parser.add_argument('--REFCYCLE_mode', type=str, default="REFREFCYCLE", choices='(CUT, cut, FastCUT, fastcut, REF, REFCYCLE)')
        parser.add_argument('--training_mode', type=str, default="four_layers", choices='(four_layers, five_layers, three_layers)')


        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_cycle', type=float, default=30.0, help='10.0 weight for cycle loss (A -> B -> A) 1.0')
        parser.add_argument('--lambda_identity_realB', type=float, default=10.0, help='weight for loss_identity_realB loss')

        
        parser.add_argument('--lambda_style_vgg_mean_variance', type=float, default=0.1, help='weight for style_vgg_mean_variance loss: style(G(X), X)')
        parser.add_argument('--lambda_style_constra', type=float, default=0.01, help='0.01weight for style_constra loss: style(G(X), X)')
        
        parser.add_argument('--lambda_content_constra', type=float, default=0.01, help='weight for content_constra loss: style(G(X), X)')
        parser.add_argument('--lambda_content_vgg', type=float, default=0.01, help='weight for content_constra loss: style(G(X), X)0.01')
        

        parser.add_argument('--use_loss_double', type=util.str2bool, nargs='?', const=True, default=True, help='use adain to the output')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()



        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        
        
        if opt.lambda_identity_realB and self.isTrain:
            self.loss_names += ['identity_A']
            self.loss_names += ['identity_B']
        
        
        # content constraints
        if opt.lambda_content_constra and self.isTrain:
            self.loss_names += ['content_constra_A']
            self.loss_names += ['content_constra_B']
        if opt.lambda_content_vgg and self.isTrain:
            self.loss_names += ['content_vgg_A']
            self.loss_names += ['content_vgg_B']
       
        
        # style constraints
        if opt.lambda_style_vgg_mean_variance and self.isTrain:
            self.loss_names += ['style_vgg_mean_variance_A']
            self.loss_names += ['style_vgg_mean_variance_B']
        if opt.lambda_style_constra and self.isTrain:
            self.loss_names += ['style_constra_A']
            self.loss_names += ['style_constra_B']
        # end add
        
        if self.opt.phase == "train" or self.opt.phase == "val":
            visual_names_A = ['real_A', 'fake_B', 'rec_A', 'idt_B']
            visual_names_B = ['real_B', 'fake_A', 'rec_B', 'idt_A']
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        else:
            self.visual_names = ['fake_B']
            
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']
            

        # define networks (both generator and discriminator)
        
        vgg = vgg19.vgg
        vgg.load_state_dict(torch.load('/ref_dehazing/wgts/vgg/vgg_normalised.pth'))
        if opt.training_mode == 'four_layers':
            vgg = nn.Sequential(*list(vgg.children())[:31])
        elif opt.training_mode == 'three_layers':
            vgg = nn.Sequential(*list(vgg.children())[:18])
        else:
            vgg = nn.Sequential(*list(vgg.children())[:44])
        self.netG_A = (networks.Net_concat(vgg, opt, opt.input_nc, opt.output_nc, opt.ngf, opt.normG)).to(opt.gpu_ids[0])
        if self.isTrain:
            self.netG_B = (networks.Net_concat(vgg, opt, opt.input_nc, opt.output_nc, opt.ngf, opt.normG)).to(opt.gpu_ids[0])


        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []



            self.criterionIdt_l1 = torch.nn.L1Loss().to(self.device)
            self.criterionIdt_mse = torch.nn.MSELoss().to(self.device)
            self.criterionCycle = torch.nn.MSELoss().to(self.device)
            
            # about the style loss
            self.criterion_style_vgg_mean_variance = Styleloss_vgg_mean_variance(opt).to(self.device)
            self.criterion_style_constra = StyleLoss_constra(opt)
            self.criterion_style_constra_1 = StyleLoss_constra_1(opt)
            self.criterion_style_constra_rhr3 = StyleLoss_constra_rhr3(opt)
            # about the content loss
            self.criterion_content_constra = ContentLoss_constra(opt).to(self.device)
            self.criterion_content_vgg = ContentLoss_vgg(opt).to(self.device)
            
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.pre.parameters(), self.netG_A.encoder_model_1.parameters(), self.netG_A.encoder_model_2.parameters(), self.netG_A.encoder_model_3.parameters(),
                                                                self.netG_A.encoder_conv1_3.parameters(), self.netG_A.res_model.parameters(), self.netG_A.decoder_model.parameters(), self.netG_A.post.parameters(),
                                                                self.netG_B.pre.parameters(), self.netG_B.encoder_model_1.parameters(), self.netG_B.encoder_model_2.parameters(), self.netG_B.encoder_model_3.parameters(),
                                                                self.netG_B.encoder_conv1_3.parameters(), self.netG_B.res_model.parameters(), self.netG_B.decoder_model.parameters(), self.netG_B.post.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_DA = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DB = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DA)
            self.optimizers.append(self.optimizer_DB)

    
    def set_input(self, input, current_epoch):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.current_epoch = current_epoch
    
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        # self.fake_B, self.idt_B, self.fake_B_feats, self.real_B_feats, self.real_A_feats, self.pseudo_label_dcp_feats, self.pseudo_label_ffa_feats  = self.netG(self.real_A, self.real_B, self.pseudo_label_dcp, self.pseudo_label_ffa)
        
        # real_A is hazy image, the real_B is reference image
        # for netG_A, the real_A is input, the real_B assists in dehazing as gudiance 
        # for netG_B, the real_B is input, the real_A assists in generating hazy image as gudiance
        
        # firstly dehazing, and then generater hazy images
        self.fake_B, self.idt_B, _, _, _  = self.netG_A(self.real_A, self.real_B, features_back = True) # G_A(A)  
            
            
    
    def forward_HRH(self):
        # firstly dehazing, and then generater hazy images
        self.fake_B, self.idt_B, self.fake_B_feats, self.real_A_feats, self.real_B_feats  = self.netG_A(self.real_A, self.real_B, features_back = True) # G_A(A)  
        if self.isTrain:
            self.rec_A, _, self.rec_A_feats = self.netG_B(self.fake_B, self.real_A) # G_B(G_A(A))
        
    
    def forward_RHR(self):
        
        #firstly generater hazy images ,and then dehazing  
        if self.isTrain:
            self.fake_A, self.idt_A, self.fake_A_feats = self.netG_B(self.real_B, self.real_A) # G_B(B)
            self.rec_B, _, self.rec_B_feats = self.netG_A(self.fake_A, self.real_B)  # G_A(G_B(B))
    
    def backward_D_basic(self, netD, real, fake, idt):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real, only_last_layer = True)
        loss_D_real = self.criterionGAN(pred_real, True).mean()
        # Fake
        pred_fake = netD(fake.detach(), only_last_layer = True)
        pred_idt = netD(idt.detach(), only_last_layer = True)
        loss_D_fake = 0.3 * self.criterionGAN(pred_fake, False).mean() + 0.7 * self.criterionGAN(pred_idt, False).mean() 
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D


    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        idt_B = self.idt_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, idt_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        # fake_A = self.fake_A_pool.query(self.fake_A)
        fake_A = self.fake_A
        idt_A = self.idt_A
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, idt_A)

    
    def backward_G_HRH(self):
        """Calculate the loss for generators G_A and G_B"""
        
        # calculate gan loss
        if self.opt.lambda_GAN > 0.0:
            self.loss_G_GAN_A = (0.7 * self.criterionGAN(self.netD_A(self.idt_B, only_last_layer = True), True).mean() + 0.3 * self.criterionGAN(self.netD_A(self.fake_B, only_last_layer = True), True).mean()) * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_A = 0.0

        
        # calculate cycle loss 
        if self.opt.lambda_cycle > 0.0:
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_cycle * sigmoid_rampup(self.current_epoch, 200) # Forward cycle loss || G_B(G_A(A)) - A||
        else:
            self.loss_cycle_A = 0.0
        
        
        # calculate identity loss
        if self.opt.lambda_identity_realB > 0:
            if self.opt.use_loss_double:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.loss_identity_A = self.opt.lambda_identity_realB * self.criterionIdt_mse(self.idt_B, self.real_B)
            else:
                self.loss_identity_A = self.opt.lambda_identity_realB * self.criterionIdt_mse(self.idt_B, self.real_B)
        else:
            self.loss_identity_A = 0.0
        
        
        # Style loss
        if self.opt.lambda_style_vgg_mean_variance > 0.0:
            if self.opt.use_loss_double:
                # Style loss A: G_A(A) VS real_B
                self.loss_style_vgg_mean_variance_A = self.opt.lambda_style_vgg_mean_variance * self.criterion_style_vgg_mean_variance(self.fake_B, self.real_B, self.fake_B_feats, self.real_B_feats, use_features=True, use_both=False)
            else:
                 # Style loss A: G_A(A) VS real_B
                self.loss_style_vgg_mean_variance_A = self.opt.lambda_style_vgg_mean_variance * self.criterion_style_vgg_mean_variance(self.fake_B, self.real_B, self.fake_B_feats, self.real_B_feats, use_features=True, use_both=False)
        else: 
            self.loss_style_vgg_mean_variance_A = 0.0
            
        
        # calculate style_constra loss
        if self.opt.lambda_style_constra > 0.0:
            fakeB = self.fake_B
            idtB = self.idt_B
            # fakeA = self.fake_A
            fake_B_featsD = self.netD_A(fakeB, only_last_layer = False)
            idt_B_featsD = self.netD_A(idtB, only_last_layer = False)
            realB_featsD_A = self.netD_A(self.real_B, only_last_layer = False)
            realA_featsD_A = self.netD_A(self.real_A, only_last_layer = False)
            
            
            if self.opt.use_loss_double:
                self.loss_style_constra_A = self.opt.lambda_style_constra * (0.3 * self.criterion_style_constra_1(self.fake_B, self.real_B, self.real_A, fake_B_featsD, realB_featsD_A, realA_featsD_A, weight_type = 'thresold_curriculum', thresold = 0.5, cl_denominator_include_pos = False) + 0.7 * self.criterion_style_constra_1(self.idt_B, self.real_B, self.real_A, idt_B_featsD, realB_featsD_A, realA_featsD_A, weight_type = 'thresold_curriculum', thresold = 0.5, cl_denominator_include_pos = False))
            else:
                self.loss_style_constra_A = self.opt.lambda_style_constra * self.criterion_style_constra_1(self.fake_B, self.real_B, self.real_A, fake_B_featsD, realB_featsD_A, realA_featsD_A, weight_type = 'thresold_curriculum', thresold = 0.5, cl_denominator_include_pos = False)
        else:
            self.loss_style_constra_A = 0.0
         
        
        
        ######################################################content loss ###########################################################
        # calculate content_constra loss
        if self.opt.lambda_content_constra > 0.0:
            if self.opt.use_loss_double:
                self.loss_content_constra_A = self.opt.lambda_content_constra * self.criterion_content_constra(self.fake_B, self.real_B, self.fake_B_feats, self.real_B_feats, weight_type = 'normalization_reverse', thresold = 0.5, cl_denominator_include_pos = False, projection_type = 'average' )
            else:
                self.loss_content_constra_A = self.opt.lambda_content_constra * self.criterion_content_constra(self.fake_B, self.reAal_B, self.fake_B_feats, self.real_B_feats, weight_type = 'normalization_reverse', thresold = 0.5, cl_denominator_include_pos = False, projection_type = 'average' )            
        else:
            self.loss_content_constra_A = 0.0

        
        
        # calculate content_vgg_relu31&relu41 loss           
        if self.opt.lambda_content_vgg > 0.0:
            if self.opt.use_loss_double:
                self.loss_content_vgg_A = self.opt.lambda_content_vgg * self.criterion_content_vgg(self.fake_B_feats, self.real_A_feats, norm = True)
            else:
                self.loss_content_vgg_A = self.opt.lambda_content_vgg * self.criterion_content_vgg(self.fake_B_feats, self.real_A_feats, norm = True)
        else:
            self.loss_content_vgg_A = 0.0  
        
        
        self.loss_G_A = self.loss_G_GAN_A + self.loss_identity_A + self.loss_content_vgg_A + self.loss_style_vgg_mean_variance_A + self.loss_style_constra_A + self.loss_content_constra_A
        self.loss_G1 = self.loss_cycle_A + self.loss_G_A
        self.loss_G1.backward()
        
        
    def backward_G_RHR(self):
        """Calculate the loss for generators G_A and G_B"""
        # calculate gan loss
        if self.opt.lambda_GAN > 0.0:
            self.loss_G_GAN_B = (0.7 * self.criterionGAN(self.netD_B(self.idt_A, only_last_layer = True), True).mean() + 0.3 * self.criterionGAN(self.netD_B(self.fake_A, only_last_layer = True), True).mean()) * self.opt.lambda_GAN 
        else:
            self.loss_G_GAN_B = 0.0
        
        
        # calculate cycle loss 
        if self.opt.lambda_cycle > 0.0:
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_cycle * sigmoid_rampup(self.current_epoch, 200) # Backward cycle loss || G_A(G_B(B)) - B||
        else:
            self.loss_cycle_B = 0.0
        
        
        # calculate identity loss
        if self.opt.lambda_identity_realB > 0:
            if self.opt.use_loss_double:
                self.loss_identity_B = self.opt.lambda_identity_realB * self.criterionIdt_mse(self.idt_A, self.real_A)
            else:
                self.loss_identity_B = 0.0
        else:
            self.loss_identity_B = 0.0

        
        
        # Style loss
        if self.opt.lambda_style_vgg_mean_variance > 0.0:
            if self.opt.use_loss_double:
                self.loss_style_vgg_mean_variance_B = self.opt.lambda_style_vgg_mean_variance * self.criterion_style_vgg_mean_variance(self.fake_A, self.real_A, self.fake_A_feats, self.real_A_feats, use_features=True, use_both=False)
            else:
                self.loss_style_vgg_mean_variance_B = 0.0
        else: 
            self.loss_style_vgg_mean_variance_B = 0.0

            

        
        # calculate style_constra loss
        if self.opt.lambda_style_constra > 0.0:
            fakeA = self.fake_A
            idtA = self.idt_A
   
            fake_A_featsD = self.netD_B(fakeA, only_last_layer = False)
            idt_A_featsD = self.netD_B(idtA, only_last_layer = False)
            realB_featsD_B = self.netD_B(self.real_B, only_last_layer = False)
            realA_featsD_B = self.netD_B(self.real_A, only_last_layer = False)
            
            if self.opt.use_loss_double:
                self.loss_style_constra_B = self.opt.lambda_style_constra * (0.3 * self.criterion_style_constra_rhr3(self.fake_A, self.real_A, self.real_B, fake_A_featsD, realA_featsD_B, realB_featsD_B, weight_type = 'thresold_curriculum', thresold = 0.5, cl_denominator_include_pos = False) + 0.7 * self.criterion_style_constra_rhr3(self.idt_A, self.real_A, self.real_B, idt_A_featsD, realA_featsD_B, realB_featsD_B, weight_type = 'thresold_curriculum', thresold = 0.5, cl_denominator_include_pos = False))
            else:
                self.loss_style_constra_B = 0.0
        else:
            self.loss_style_constra_B = 0.0            
        
        
        ######################################################content loss ###########################################################

        # calculate content_constra loss
        if self.opt.lambda_content_constra > 0.0:
            if self.opt.use_loss_double:
                self.loss_content_constra_B = self.opt.lambda_content_constra * self.criterion_content_constra(self.fake_A, self.real_A, self.fake_A_feats, self.real_A_feats, weight_type = 'normalization_reverse', thresold = 0.5, cl_denominator_include_pos = False, projection_type = 'average' )
            else:         
                self.loss_content_constra_B = 0.0
        else:
            self.loss_content_constra_B = 0.0
        
        
        # calculate content_vgg_relu31&relu41 loss           
        if self.opt.lambda_content_vgg > 0.0:
            if self.opt.use_loss_double:
                self.loss_content_vgg_B = self.opt.lambda_content_vgg * self.criterion_content_vgg(self.fake_A_feats, self.real_B_feats, norm = True)
            else:
                self.loss_content_vgg_B = 0.0
        else:
            self.loss_content_vgg_B = 0.0
            
        
        self.loss_G_B = self.loss_G_GAN_B + self.loss_identity_B + self.loss_content_vgg_B + self.loss_style_vgg_mean_variance_B + self.loss_style_constra_B + self.loss_content_constra_B 
        self.loss_G2 = self.loss_cycle_B + self.loss_G_B
        self.loss_G2.backward()
    

    
    def data_dependent_initialize(self, data, epoch):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        # bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        bs_per_gpu = 1
        # print('bs_per_gpu:',bs_per_gpu)
        self.set_input(data, epoch)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:                
            
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.backward_G()          # calculate graidents for G
            
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF_A.parameters(), self.netF_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)
    
    
    
    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals
    
    
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        
        # forward
        # self.forward()      # compute fake images and reconstruction images.
        self.forward_HRH()
        
        # D_A and D_B
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_DA.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_DA.step()  # update D_A and D_B's weights

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G_HRH()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights   

        if self.opt.isTrain:
            self.forward_RHR()
            self.set_requires_grad(self.netD_B, True)
            self.optimizer_DB.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_DB.step()  # update D_A and D_B's weights
            
            
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G_RHR()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights        


