import torch
import torch.nn as nn
# import imagehash
from torch.nn.functional import l1_loss
from torch.autograd import Variable
import pyiqa


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std2(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat
    
       

class ContentLoss_vgg(nn.Module):
    def __init__(self, opt):
        super(ContentLoss_vgg, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.weights_layers = [1.0, 1.0, 1.0, 1.0, 0.5]
        self.weights_layers = [1.0, 1.0, 1.0, 1.0, 1.0] # relu11-relu51

    def forward(self, dehaze_features, haze_features, norm = True):
        loss_c = 0
        dehaze_feats = dehaze_features[1:]
        haze_feats = haze_features[1:]
        if(norm == False):
            for id in range(2, len(dehaze_feats)):
                loss_c += self.weights_layers[id] * self.mse_loss(dehaze_feats[id], haze_feats[id])
        else:
            for id in range(2, len(dehaze_feats)):
                 loss_c += self.weights_layers[id] * self.mse_loss(mean_variance_norm(dehaze_feats[id]), mean_variance_norm(haze_feats[id]))
        return loss_c
        

        
        
class Styleloss_vgg_mean_variance(nn.Module):
    def __init__(self, opt):
        super(Styleloss_vgg_mean_variance, self).__init__()
        self.mse_loss = nn.MSELoss()

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std2(input)
        target_mean, target_std = calc_mean_std2(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, dehaze, ref, dehaze_feats, ref_feats, use_features=False, use_both=False):
        loss_s = 0
        dehaze_feats = dehaze_feats[1:]
        ref_feats = ref_feats[1:]
        if use_both:
            loss_s = self.calc_style_loss(dehaze, ref)
            for c, s in zip(dehaze_feats, ref_feats):
                loss_s += self.calc_style_loss(c, s)
        else:
            if use_features: 
                for c, s in zip(dehaze_feats, ref_feats):
                    loss_s += self.calc_style_loss(c, s)
            else:
                loss_s = self.calc_style_loss(dehaze, ref)
        return loss_s
        



class GaussianHistogram(nn.Module):
    """
    Use gaussian distribution
    Args:
        bins: number of bins to seperate values
        min: minium vale of the data
        max: maximum value of the data
        sigma: a learable paramerter, init=0.01
    """

    def __init__(self, bins, min, max, sigma, require_grad=False):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max

        self.sigma = torch.tensor([sigma])
        self.sigma = Variable(self.sigma, requires_grad=require_grad)

        self.delta = float(max - min) / float(bins)
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

    def forward(self, x, attention_mask=None):
        device = x.device
        self.sigma = self.sigma.to(device)
        self.centers = self.centers.to(device)

        x = torch.unsqueeze(x, dim=1) - torch.unsqueeze(self.centers, 1)
        hist_dist = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        # multiply with attention mask
        if not type(attention_mask) == type(None):
            hist_dist *= torch.unsqueeze(attention_mask, 1)

        hist = hist_dist.sum(dim=-1)
        hist = hist / torch.sum(hist, dim=1, keepdim=True)

        return hist, hist_dist


class EarthMoverDisteLoss(nn.Module):
    """
    Earth Mover Distance Loss
    Args:
    """
    def __init__(self):
        super(EarthMoverDisteLoss, self).__init__()
        self.creterion = nn.MSELoss()

    def forward(self, input, target):
        """
        Loss calculation
        :param input:  input histogram, shape required: (N, K)
        :param target: target histogram, shape required: (N, K)
        :return:
        """
        input_cumsum = torch.cumsum(input, dim=1)
        target_cumsum = torch.cumsum(target, dim=1)
        return self.creterion(input_cumsum, target_cumsum)


class HistogramLoss(nn.Module):
    """
    Calculate histogram distribution loss #TODO: Make RGB also avaialble, right now only yuv supported
    Args:
    """
    def __init__(self, opt):
        super(HistogramLoss, self).__init__()
        self.creterion = EarthMoverDisteLoss()
        self.histlayer = GaussianHistogram(bins=256, min=0, max=1, sigma=0.01)

    def forward(self, input, target):
        channels = input.shape[1]
        losses = []
        for i in range(channels):
            input_channel = torch.flatten(input[:, i, :, :], start_dim=1, end_dim=-1)
            target_channel = torch.flatten(target[:, i, :, :], start_dim=1, end_dim=-1)

            input_hist, _ = self.histlayer(input_channel)
            target_hist, _ = self.histlayer(target_channel)
            losses.append(self.creterion(input_hist, target_hist))

        return sum(losses)




def Contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.
    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).
      
    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    N, C, H, W = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)

    d = 1 - cosine_sim                                  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
    d_min, _ = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)

    # Eq (2)
    d_tilde = d / (d_min + 1e-5)
    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)
    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)
    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))
    
    return cx_loss
    
    


class ContextualLoss(nn.Module):

    def __init__(self, opt):
        super(ContextualLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def contextual_loss(self, x, y, h=0.5):
        """Computes contextual loss between x and y.
        Args:
          x: features of shape (N, C, H, W).
          y: features of shape (N, C, H, W).
          
        Returns:
          cx_loss = contextual loss between x and y (Eq (1) in the paper)
        """
        assert x.size() == y.size()
        N, C, H, W = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

        y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

        # The equation at the bottom of page 6 in the paper
        # Vectorized computation of cosine similarity for each pair of x_i and y_j
        x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)

        d = 1 - cosine_sim                                  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
        d_min, _ = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)

        # Eq (2)
        d_tilde = d / (d_min + 1e-5)
        # Eq(3)
        w = torch.exp((1 - d_tilde) / h)
        # Eq(4)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)
        # Eq (1)
        cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
        cx_loss = torch.mean(-torch.log(cx + 1e-5))
        return cx_loss

    def forward(self, pre, reference):
        loss = self.contextual_loss(pre, reference)
        return loss





class StyleLoss_gram(nn.Module):

    def __init__(self, opt):
        super(StyleLoss_gram, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def gram_matrix(self, input):
        b, c, h, w= input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(b, c, h * w)  # resize F_XL into \hat F_XL
        features_t = features.transpose(1, 2)
        # we compute the gram product and 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps
        gram = torch.bmm(features, features_t) / (c * h * w)  
        return gram

    def forward(self, pre_freatures, reference_freatures):
       
        style_loss = 0
        for a, b in zip(pre_freatures, reference_freatures):
            pre_gram = self.gram_matrix(a)
            target_gram = self.gram_matrix(b)
            style_loss += self.mse_loss(pre_gram, target_gram)
        
        return style_loss




projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128)
)

projection_content = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128)
)




class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)



class ContentLoss_constra(nn.Module):
    def __init__(self, opt):
    # def __init__(self):
        super(ContentLoss_constra, self).__init__()
        # self.vgg_encoder = VGGEncoder()
        
        self.proj_style = projection_style
        self.proj_content = projection_content
        
        # Projection 
        # self.feature_projection = torch.nn.AdaptiveAvgPool2d((2, 2))
                                
        self.feature_projection_averagepool = nn.Sequential(
                                torch.nn.AdaptiveAvgPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        self.feature_projection_maxpool = nn.Sequential(
                                torch.nn.AdaptiveMaxPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # if opt.training_mode == 'four_layers':
            # self.weights_layers = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # elif opt.training_mode == 'five_layers':
            # self.weights_layers = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4]
        
        self.weights_layers = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.weights_layers = [1.0, 1.0, 1.0, 1.0, 1.0]
   
    def norm(self, input):
        out = input / torch.norm(input, p=2, dim=1, keepdim=True)
        return out

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        #loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        return loss
        
    def compute_contrastive_loss_weight_bmm(self, feat_q, feat_k, index, weight, tau):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        # print("out:", out.shape)
        # print("weight", weight.shape)
        loss = out.squeeze(0)[index] / torch.sum(out.squeeze(0) * weight)
        return loss
    
    def compute_contrastive_loss_weight_bmm1(self, feat_q, feat_k, index, weight, tau):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        out = out.squeeze(0)
        out = torch.exp(out[index]) / torch.sum(torch.exp(out) * weight)
        loss = -torch.log(out)
        return loss
    
    def compute_contrastive_loss_weight_l1(self, feat_q, feat_k, tau, index, weight):
        l1_loss = torch.nn.L1Loss(reduction = 'none')
        out = torch.mean(l1_loss(feat_q.repeat(feat_k.shape[0],1) , feat_k), dim = 1)# torch.Size([B, C*H*W]) = torch.Size([B])
        loss = out[index] / torch.sum(out * weight)
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def content_feature_contrastive(self, input):
        #out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out
            
       
    def forward(self, out, ref_images, out_feats, ref_feats, weight_type = 'normalization', thresold = 0.5, cl_denominator_include_pos = False, projection_type = 'average'):
    
        # similarity between the coresponding reference image and the other reference images
        b, c, h, w= out.size()
        # out1 = out.flatten(2, 3) #(b, c, h*w)
        ref_images_copy = ref_images.flatten(2, 3) #(b, c, h*w)
        similarity = []
        weight = []
        for i in range(b):
            ref_images_current = ref_images_copy[i,:,:].unsqueeze(0).repeat(b,1,1) 
            similarity_current = torch.mean(self.cos(ref_images_current, ref_images_copy), dim=1)  # tensor[b, h*w] to tensor[b]
            similarity_current[i] = 0 #setting the itself's similarty as zero
            similarity.append(similarity_current)

            
            if weight_type == 'normalization':
                weight_current = similarity_current / torch.sum(similarity_current)
            elif weight_type == 'normalization_reverse':
                weight_current = 1.0 - (similarity_current / torch.sum(similarity_current))
            elif weight_type == 'top_k': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                _, indices = torch.sort(similarity_current, descending=True)
                weight_current = torch.where(indices > int(b/2), (1 - thresold) * torch.ones_like(similarity_current), (1 + thresold) * torch.ones_like(similarity_current))
            elif weight_type == 'thresold': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                similarity_mean = torch.sum(similarity_current)/ (b-1)
                weight_current = torch.where(similarity_current > similarity_mean, (1 - thresold) * torch.ones_like(similarity_current), (1 + thresold) * torch.ones_like(similarity_current))
            elif weight_type == 'thresold_top_k':
                 _, indices = torch.sort(similarity_current, descending=True)
                 similarity_mean = torch.sum(similarity_current)/ (b-1)
                 weight_current = torch.where(similarity_current > similarity_mean, (1 - thresold) * torch.ones_like(similarity_current), (1 + thresold) * torch.ones_like(similarity_current))
                 weight_current = torch.where(indices > int(b/2), weight_current, (1 + thresold) * torch.ones_like(similarity_current))
            else:
                 weight_current = torch.ones_like(similarity_current) / (b-1)
                             
            
            if cl_denominator_include_pos:
                weight_current[i] = 1
            else: 
                weight_current[i] = 0
            # print('weight_current:', weight_current.shape)
            weight.append(weight_current)
         
                
        # ref is positive ; input is negtive
        # output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        # ref_middle_features = self.vgg_encoder(ref_images, output_last_feature=False)
        # input_middle_features = self.vgg_encoder(input, output_last_feature=False)
        loss = 0
        for id in range(2, len(out_feats)):

           if projection_type == 'average':
               ref_feat = self.feature_projection_averagepool(ref_feats[id])
               out_feat = self.feature_projection_averagepool(out_feats[id])
           elif projection_type == 'max':
               ref_feat = self.feature_projection_maxpool(ref_feats[id])
               out_feat = self.feature_projection_maxpool(out_feats[id])
           elif projection_type == 'average_max':
               ref_feat1 = self.feature_projection_averagepool(ref_feats[id])
               out_feat1 = self.feature_projection_averagepool(out_feats[id])
               ref_feat2 = self.feature_projection_maxpool(ref_feats[id])
               out_feat2 = self.feature_projection_maxpool(out_feats[id])
               ref_feat =  torch.cat([ref_feat1, ref_feat2], 1)
               out_feat =  torch.cat([out_feat1, out_feat2], 1)
           
           ref_feat = self.norm(ref_feat) # torch.Size([B, C*H*W])
           out_feat = self.norm(out_feat) # torch.Size([B, C*H*W])
           # print(in_feature.shape) #  torch.Size([2, 256, 2, 2]) to  torch.Size([2, 1024])
           
           content_contrastive_loss = 0
           for i in range(out_feat.shape[0]):
                wgt = weight[i]
                # print('wgt:', wgt)
                # print(style_feature[i].shape)
                anchor = out_feat[i].unsqueeze(0) # torch.Size([1, C*H*W])
                content_contrastive_loss += self.compute_contrastive_loss_weight_bmm1(anchor, ref_feat, i, wgt, tau=0.2)
           loss += (content_contrastive_loss/b) * self.weights_layers[id-1]
        
        return loss 



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]



class StyleLoss_constra(nn.Module):
    def __init__(self, opt):
        super(StyleLoss_constra, self).__init__()
        # self.vgg = Vgg19().to(opt.gpu_ids[0])
        
        self.proj_style = projection_style
        self.proj_content = projection_content
        
                                
        self.feature_projection_averagepool = nn.Sequential(
                                torch.nn.AdaptiveAvgPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        self.feature_projection_maxpool = nn.Sequential(
                                torch.nn.AdaptiveMaxPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_1 = nn.MSELoss(reduction='none')
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # self.iqa_metric = pyiqa.create_metric('musiq', as_loss=True).to(opt.gpu_ids[0])
        # self.iqa_metric = pyiqa.create_metric('musiq', as_loss=True).cuda()
        # self.weights_layers = [1.0, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32]
        # self.weights_layers = [1.0, 1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 16]
        # self.weights_layers = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights_layers = [1.0, 1.0, 1.0, 1.0, 1.0] # relu11-relu51
        
    def iqa_metric(self, device):
        return pyiqa.create_metric('musiq', as_loss=True).to(device)
    
    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)   #C和w*h转置
        gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t  ,and then normalize' the values of the gram matrix
        return gram
    
    def norm(self, input):
        out = input / torch.norm(input, p=2, dim=1, keepdim=True)
        return out
    
    def forward(self, out, ref, haze, out_images_featsD, ref_images_featsD, haze_images_featsD, weight_type = 'normalization', thresold = 0.5, cl_denominator_include_pos = False, projection_type = 'average'):
        #self.fake_B, self.real_B, self.real_A, fake_B_featsD, real_B_featsD, self.real_A_featsD
        
        # MUSIQ值越大，表示图像质量越好,在这个里面，我们都认为清晰图是清晰的，但是雾图中可能存在错误的。
        # for the dehazing network, dehaze, clear(pos), haze(neg),
        # for the rehazing network, rehaze, haze(pos), clear(neg)
        
        
        # n1_vgg = self.vgg(n1)
        # n1_vgg = self.vgg(n1)
        # n2_vgg = self.vgg(n2)
        # n3_vgg = self.vgg(n3)
        # n4_vgg = self.vgg(n4)
        # out_vgg = self.vgg(out)
        # haze_vgg = self.vgg(haze)
        # ref_vgg = self.vgg(ref_images)
        
        iqa_metric = pyiqa.create_metric('musiq', as_loss=True).to(haze.device)
        
        
        # n1 = n1_feats[0]
        # n2 = n2_feats[0]
        # n3 = n3_feats[0]
        # n4 = n4_feats[0]
        # out = out_feats[0] 
        # ref = ref_images_feats[0]
        # haze = haze_feats[0]
        
        
        # n1_vgg = n1_feats[1:]
        # n2_vgg = n2_feats[1:]
        # n3_vgg = n3_feats[1:]
        # n4_vgg = n4_feats[1:]
        # out_vgg = out_feats[1:]
        # ref_vgg = ref_images_feats[1:] 
        # haze_vgg = haze_feats[1:]
        
        
        out_vgg = out_images_featsD
        ref_vgg = ref_images_featsD
        haze_vgg = haze_images_featsD
        
        loss_all = 0
        
        batch_size = out.shape[0]
        
        musiq = []
        for j in range(out.shape[0]):
            musiq_haze = iqa_metric(haze[j,:,:,:])
            musiq.append(musiq_haze)
        musiq = torch.tensor(musiq)
        
        
        for id in range(out.shape[0]):
            weight = []
            # musiq_haze = self.iqa_metric(haze[id,:,:,:])
            # musiq.append(musiq_haze)
            # musiq_n1 = self.iqa_metric(n1[id,:,:,:])
            # musiq.append(musiq_n1)
            # musiq_n2 = self.iqa_metric(n2[id,:,:,:])
            # musiq.append(musiq_n2)
            # musiq_n3 = self.iqa_metric(n3[id,:,:,:])
            # musiq.append(musiq_n3)
            # musiq_n4 = self.iqa_metric(n4[id,:,:,:])
            # musiq.append(musiq_n4)

            # musiq_haze = (self.iqa_metric(haze[id,:,:,:].device))(haze[id,:,:,:])
            # musiq.append(musiq_haze)
            # musiq_n1 = (self.iqa_metric(haze[id,:,:,:].device))(n1[id,:,:,:])
            # musiq.append(musiq_n1)
            # musiq_n2 = (self.iqa_metric(haze[id,:,:,:].device))(n2[id,:,:,:])
            # musiq.append(musiq_n2)
            

            if weight_type == 'normalization':
                weight = musiq / torch.sum(musiq)
            elif weight_type == 'normalization_reverse':
                weight = 1.0 - (musiq / torch.sum(musiq))
            elif weight_type == 'top_k': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                _, indices = torch.sort(musiq, descending=True)
                weight = torch.where(indices > int((musiq.shape[0])/2), (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = torch.mean(musiq)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold_1': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = torch.mean(musiq)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                weight = weight / torch.sum(weight)
            elif weight_type == 'thresold_curriculum': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                # weight = weight / torch.sum(weight)
            elif weight_type == 'thresold_curriculum_1': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 + thresold) * torch.ones_like(musiq), 1.0 * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_2': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), 1.0 * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_3': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 + thresold) * torch.ones_like(musiq), (1 - thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_4': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, 1.0 * torch.ones_like(musiq), (1 + thresold)  * torch.ones_like(musiq))
            elif weight_type == 'thresold_top_k':
                 _, indices = torch.sort(musiq, descending=True)
                 musiq_mean = torch.mean(musiq)
                 weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                 weight = torch.where(indices > int((musiq.shape[0])/2), weight, (1 + thresold) * torch.ones_like(musiq))
            else:
                 # weight = torch.ones_like(musiq)/ (musiq.shape[0])
                 weight = torch.ones_like(musiq)
             
      
            loss = 0
            # for i in range(len(out_vgg)):
            for i in range(len(out_vgg)-1, len(out_vgg)):

                
                loss_pos_current = 0
                loss_neg_current = 0
                loss_current = 0
                
                
                if projection_type == 'average':
                    ref_feat = self.feature_projection_averagepool(ref_vgg[i])
                    out_feat = self.feature_projection_averagepool(out_vgg[i])
                    haze_feat = self.feature_projection_averagepool(haze_vgg[i])
                elif projection_type == 'max':
                    ref_feat = self.feature_projection_maxpool(ref_vgg[i])
                    out_feat = self.feature_projection_maxpool(out_vgg[i])
                    haze_feat = self.feature_projection_maxpool(haze_vgg[i])
                elif projection_type == 'average_max':
                    ref_feat1 = self.feature_projection_averagepool(ref_vgg[i])
                    out_feat1 = self.feature_projection_averagepool(out_vgg[i])
                    haze_feat1 = self.feature_projection_averagepool(haze_vgg[i])
                    ref_feat2 = self.feature_projection_maxpool(ref_vgg[i])
                    out_feat2 = self.feature_projection_maxpool(out_vgg[i])
                    haze_feat2 = self.feature_projection_maxpool(haze_vgg[i])
                    ref_feat =  torch.cat([ref_feat1, ref_feat2], 1)
                    out_feat =  torch.cat([out_feat1, out_feat2], 1)
                    haze_feat =  torch.cat([haze_feat1, haze_feat2], 1)
               
                # ref_feat = self.norm(ref_feat) # torch.Size([B, C*H*W])
                # out_feat = self.norm(out_feat) # torch.Size([B, C*H*W])
                # haze_feat = self.norm(haze_feat)
                
                
                
                out_feat_copy = out_feat[id].unsqueeze(0).repeat(batch_size, 1) 
                loss_neg_current = weight.to(out_feat_copy.device) * torch.mean(self.mse_loss_1(out_feat_copy, haze_feat), dim=1)

                
                # loss_pos_current = self.mse_loss(out_feat[id], ref_feat[id].detach())
                loss_pos_current = torch.mean(self.mse_loss_1(out_feat_copy, ref_feat.detach()), dim=1)
                
                loss_current = (torch.sum(loss_pos_current)) / (torch.sum(loss_neg_current) + 1e-7)
                loss += self.weights_layers[i] * loss_current
            
            loss_all += (loss / out.shape[0])
        return loss_all 
            




class StyleLoss_constra_1(nn.Module):
    def __init__(self, opt):
        super(StyleLoss_constra_1, self).__init__()
        # self.vgg = Vgg19().to(opt.gpu_ids[0])
        
        self.proj_style = projection_style
        self.proj_content = projection_content
        
                                
        self.feature_projection_averagepool = nn.Sequential(
                                torch.nn.AdaptiveAvgPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        self.feature_projection_maxpool = nn.Sequential(
                                torch.nn.AdaptiveMaxPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_1 = nn.MSELoss(reduction='none')
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        

        self.weights_layers = [1.0, 1.0, 1.0, 1.0, 1.0] # relu11-relu51
        
    def iqa_metric(self, device):
        return pyiqa.create_metric('musiq', as_loss=True).to(device)
    
    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)   #C和w*h转置
        gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t  ,and then normalize' the values of the gram matrix
        return gram
    
    def norm(self, input):
        out = input / torch.norm(input, p=2, dim=1, keepdim=True)
        return out
    
    def forward(self, out, ref, haze, out_images_featsD, ref_images_featsD, haze_images_featsD, weight_type = 'normalization', thresold = 0.5, cl_denominator_include_pos = False, projection_type = 'average'):
        #self.fake_B, self.real_B, self.real_A, fake_B_featsD, real_B_featsD, self.real_A_featsD
        
        # MUSIQ值越大，表示图像质量越好,在这个里面，我们都认为清晰图是清晰的，但是雾图中可能存在错误的。
        # for the dehazing network, dehaze, clear(pos), haze(neg),
        # for the rehazing network, rehaze, haze(pos), clear(neg)  
        
        
        iqa_metric = pyiqa.create_metric('musiq', as_loss=True).to(haze.device)
        
        
        
        out_vgg = out_images_featsD
        ref_vgg = ref_images_featsD
        haze_vgg = haze_images_featsD
        
        loss_all = 0
        batch_size = out.shape[0]
        
        musiq_pos = []
        musiq_neg = []
        for j in range(out.shape[0]):
            musiq_ref = iqa_metric(ref[j,:,:,:])
            musiq_pos.append(musiq_ref)
        musiq_pos = torch.tensor(musiq_pos)  # the musiq lower, the hard positive
        
        for j in range(out.shape[0]):
            musiq_haze = iqa_metric(haze[j,:,:,:])
            musiq_neg.append(musiq_haze)
        musiq_neg = torch.tensor(musiq_neg)
        
        
        for id in range(out.shape[0]):
            weight = []
            
            if weight_type == 'normalization':
                weight = musiq / torch.sum(musiq)
            elif weight_type == 'normalization_reverse':
                weight = 1.0 - (musiq / torch.sum(musiq))
            elif weight_type == 'top_k': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                _, indices = torch.sort(musiq, descending=True)
                weight = torch.where(indices > int((musiq.shape[0])/2), (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = torch.mean(musiq)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold_1': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = torch.mean(musiq)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                weight = weight / torch.sum(weight)
            elif weight_type == 'thresold_curriculum': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq_neg.device)
                weight_neg = torch.where(musiq_neg > musiq_mean, (1 - thresold) * torch.ones_like(musiq_neg), (1 + thresold) * torch.ones_like(musiq_neg))
                weight_pos = torch.where(musiq_pos < musiq_mean, (1 - thresold) * torch.ones_like(musiq_neg), (1 + thresold) * torch.ones_like(musiq_neg))
                # weight = weight / torch.sum(weight)
            elif weight_type == 'thresold_curriculum_1': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 + thresold) * torch.ones_like(musiq), 1.0 * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_2': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), 1.0 * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_3': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 + thresold) * torch.ones_like(musiq), (1 - thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_4': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, 1.0 * torch.ones_like(musiq), (1 + thresold)  * torch.ones_like(musiq))
            elif weight_type == 'thresold_top_k':
                 _, indices = torch.sort(musiq, descending=True)
                 musiq_mean = torch.mean(musiq)
                 weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                 weight = torch.where(indices > int((musiq.shape[0])/2), weight, (1 + thresold) * torch.ones_like(musiq))
            else:
                 # weight = torch.ones_like(musiq)/ (musiq.shape[0])
                 weight = torch.ones_like(musiq)
             
      
            loss = 0
            # for i in range(len(out_vgg)):
            for i in range(len(out_vgg)-1, len(out_vgg)):

                
                loss_pos_current = 0
                loss_neg_current = 0
                loss_current = 0
                
                
                if projection_type == 'average':
                    ref_feat = self.feature_projection_averagepool(ref_vgg[i])
                    out_feat = self.feature_projection_averagepool(out_vgg[i])
                    haze_feat = self.feature_projection_averagepool(haze_vgg[i])
                elif projection_type == 'max':
                    ref_feat = self.feature_projection_maxpool(ref_vgg[i])
                    out_feat = self.feature_projection_maxpool(out_vgg[i])
                    haze_feat = self.feature_projection_maxpool(haze_vgg[i])
                elif projection_type == 'average_max':
                    ref_feat1 = self.feature_projection_averagepool(ref_vgg[i])
                    out_feat1 = self.feature_projection_averagepool(out_vgg[i])
                    haze_feat1 = self.feature_projection_averagepool(haze_vgg[i])
                    ref_feat2 = self.feature_projection_maxpool(ref_vgg[i])
                    out_feat2 = self.feature_projection_maxpool(out_vgg[i])
                    haze_feat2 = self.feature_projection_maxpool(haze_vgg[i])
                    ref_feat =  torch.cat([ref_feat1, ref_feat2], 1)
                    out_feat =  torch.cat([out_feat1, out_feat2], 1)
                    haze_feat =  torch.cat([haze_feat1, haze_feat2], 1)
                
                
                
                out_feat_copy = out_feat[id].unsqueeze(0).repeat(batch_size, 1) 
                loss_neg_current = weight_neg.to(out_feat_copy.device) * torch.mean(self.mse_loss_1(out_feat_copy, haze_feat), dim=1)

                
                # loss_pos_current = self.mse_loss(out_feat[id], ref_feat[id].detach())
                # loss_pos_current = torch.mean(self.mse_loss_1(out_feat_copy, ref_feat.detach()), dim=1)
                loss_pos_current = weight_pos.to(out_feat_copy.device) * torch.mean(self.mse_loss_1(out_feat_copy, ref_feat), dim=1)
                
                loss_current = (torch.sum(loss_pos_current)) / (torch.sum(loss_neg_current) + 1e-7)
                loss += self.weights_layers[i] * loss_current
            
            loss_all += (loss / out.shape[0])
        return loss_all           
            
            


class StyleLoss_constra_rhr3(nn.Module):
    def __init__(self, opt):
        super(StyleLoss_constra_rhr3, self).__init__()
        # self.vgg = Vgg19().to(opt.gpu_ids[0])
        
        self.proj_style = projection_style
        self.proj_content = projection_content
        
                                
        self.feature_projection_averagepool = nn.Sequential(
                                torch.nn.AdaptiveAvgPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        self.feature_projection_maxpool = nn.Sequential(
                                torch.nn.AdaptiveMaxPool2d((2, 2)), 
                                torch.nn.Flatten(start_dim=1, end_dim=-1))
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_1 = nn.MSELoss(reduction='none')
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        

        self.weights_layers = [1.0, 1.0, 1.0, 1.0, 1.0] # relu11-relu51
        
    def iqa_metric(self, device):
        return pyiqa.create_metric('musiq', as_loss=True).to(device)
    
    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)   #C和w*h转置
        gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t  ,and then normalize' the values of the gram matrix
        return gram
    
    def norm(self, input):
        out = input / torch.norm(input, p=2, dim=1, keepdim=True)
        return out
    
    def forward(self, out, haze, ref, out_images_featsD, haze_images_featsD, ref_images_featsD, weight_type = 'normalization', thresold = 0.5, cl_denominator_include_pos = False, projection_type = 'average'):
        #self.fake_B, self.real_B, self.real_A, fake_B_featsD, real_B_featsD, self.real_A_featsD
        
        # MUSIQ值越大，表示图像质量越好,在这个里面，我们都认为清晰图是清晰的，但是雾图中可能存在错误的。
        # for the dehazing network, dehaze, clear(pos), haze(neg),
        # for the rehazing network, rehaze, haze(pos), clear(neg)  
        
        
        iqa_metric = pyiqa.create_metric('musiq', as_loss=True).to(haze.device)
        
        
        
        out_vgg = out_images_featsD
        ref_vgg = ref_images_featsD
        haze_vgg = haze_images_featsD
        
        loss_all = 0
        batch_size = out.shape[0]
        
        musiq_pos = []
        musiq_neg = []
        for j in range(out.shape[0]):
            musiq_ref = iqa_metric(ref[j,:,:,:])
            musiq_neg.append(musiq_ref)
        musiq_neg = torch.tensor(musiq_neg)  # the musiq lower, the hard positive
        
        for j in range(out.shape[0]):
            musiq_haze = iqa_metric(haze[j,:,:,:])
            musiq_pos.append(musiq_haze)
        musiq_pos = torch.tensor(musiq_pos)
        
        
        for id in range(out.shape[0]):
            weight = []
            
            if weight_type == 'normalization':
                weight = musiq / torch.sum(musiq)
            elif weight_type == 'normalization_reverse':
                weight = 1.0 - (musiq / torch.sum(musiq))
            elif weight_type == 'top_k': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                _, indices = torch.sort(musiq, descending=True)
                weight = torch.where(indices > int((musiq.shape[0])/2), (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = torch.mean(musiq)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold_1': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = torch.mean(musiq)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                weight = weight / torch.sum(weight)
            elif weight_type == 'thresold_curriculum': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq_neg.device)
                # weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                
                weight_neg = torch.where(musiq_neg < musiq_mean, (1 - thresold) * torch.ones_like(musiq_neg), (1 + thresold) * torch.ones_like(musiq_neg))
                weight_pos = torch.where(musiq_pos > musiq_mean, (1 - thresold) * torch.ones_like(musiq_neg), (1 + thresold) * torch.ones_like(musiq_neg))
                # hazy is positive samples , the larger musiq of hazy , the hard,  没啥雾，应该权重小
                # ref is negative samples , the larger musiq of hazy , the easy,  没啥雾，应该权重大
                # weight = weight / torch.sum(weight)
            elif weight_type == 'thresold_curriculum_1': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 + thresold) * torch.ones_like(musiq), 1.0 * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_2': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), 1.0 * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_3': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, (1 + thresold) * torch.ones_like(musiq), (1 - thresold) * torch.ones_like(musiq))
            elif weight_type == 'thresold_curriculum_4': # larger than mean_value is setting as 1-lambda (hard negative samples), while smaller than mean_value is setting as 1+lambda(easy negative samples)
                musiq_mean = iqa_metric(out[id,:,:,:]).to(musiq.device)
                weight = torch.where(musiq > musiq_mean, 1.0 * torch.ones_like(musiq), (1 + thresold)  * torch.ones_like(musiq))
            elif weight_type == 'thresold_top_k':
                 _, indices = torch.sort(musiq, descending=True)
                 musiq_mean = torch.mean(musiq)
                 weight = torch.where(musiq > musiq_mean, (1 - thresold) * torch.ones_like(musiq), (1 + thresold) * torch.ones_like(musiq))
                 weight = torch.where(indices > int((musiq.shape[0])/2), weight, (1 + thresold) * torch.ones_like(musiq))
            else:
                 # weight = torch.ones_like(musiq)/ (musiq.shape[0])
                 weight = torch.ones_like(musiq)
             
      
            loss = 0
            # for i in range(len(out_vgg)):
            for i in range(len(out_vgg)-1, len(out_vgg)):

                
                loss_pos_current = 0
                loss_neg_current = 0
                loss_current = 0
                
                
                if projection_type == 'average':
                    ref_feat = self.feature_projection_averagepool(ref_vgg[i])
                    out_feat = self.feature_projection_averagepool(out_vgg[i])
                    haze_feat = self.feature_projection_averagepool(haze_vgg[i])
                elif projection_type == 'max':
                    ref_feat = self.feature_projection_maxpool(ref_vgg[i])
                    out_feat = self.feature_projection_maxpool(out_vgg[i])
                    haze_feat = self.feature_projection_maxpool(haze_vgg[i])
                elif projection_type == 'average_max':
                    ref_feat1 = self.feature_projection_averagepool(ref_vgg[i])
                    out_feat1 = self.feature_projection_averagepool(out_vgg[i])
                    haze_feat1 = self.feature_projection_averagepool(haze_vgg[i])
                    ref_feat2 = self.feature_projection_maxpool(ref_vgg[i])
                    out_feat2 = self.feature_projection_maxpool(out_vgg[i])
                    haze_feat2 = self.feature_projection_maxpool(haze_vgg[i])
                    ref_feat =  torch.cat([ref_feat1, ref_feat2], 1)
                    out_feat =  torch.cat([out_feat1, out_feat2], 1)
                    haze_feat =  torch.cat([haze_feat1, haze_feat2], 1)
               
                
                
                out_feat_copy = out_feat[id].unsqueeze(0).repeat(batch_size, 1) 
                loss_neg_current = weight_neg.to(out_feat_copy.device) * torch.mean(self.mse_loss_1(out_feat_copy, ref_feat), dim=1)

                
                # loss_pos_current = self.mse_loss(out_feat[id], ref_feat[id].detach())
                # loss_pos_current = torch.mean(self.mse_loss_1(out_feat_copy, ref_feat.detach()), dim=1)
                loss_pos_current = weight_pos.to(out_feat_copy.device) * torch.mean(self.mse_loss_1(out_feat_copy, haze_feat.detach()), dim=1)
                
                loss_current = (torch.sum(loss_pos_current)) / (torch.sum(loss_neg_current) + 1e-7)
                loss += self.weights_layers[i] * loss_current
            
            loss_all += (loss / out.shape[0])
        return loss_all        







# if __name__ == '__main__':
    # data = 1.0 * torch.ones((2, 256, 400))
    # mean = torch.ones(2, 1, 400) * 2.3
    # t = data * mean
    # print(data - mean)
    
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # input1 = torch.randn(3, 64, 10, 12)
    # input2 = torch.randn(3, 64, 10, 12)
    
    # print(input2[1,:,:,:].shape)
    
    # input1 = input1.flatten(2, 3)
    # input2 = input2.flatten(2, 3)
    # l1_loss = nn.L1Loss(size_average=False, reduce=False)
    
    # output = cos(input1, input2)
    # output1 = l1_loss(input1, input2)
    
    # output[0, 0, 0]
    # print(output[0, 0, 0])
    # print(output.shape)
    # print(output1.shape)
    # print(torch.mean(output,dim=1).shape)
    
    
    # parser.add_argument('--lambda', type=float, default=3.0)
    # opt, _ = parser.parse_known_args()
    # algorithm = ContentLoss_constra()
    # out = torch.randn(2, 3, 10, 12)
    # ref_images = torch.randn(2, 3, 10, 12)
    
    # out_feats = torch.randn(2, 64, 10, 12)
    # ref_feats = torch.randn(3, 64, 10, 12)
    # out = algorithm(out, ref_images, out_feats, ref_feats, weight_type = 'normalization', cl_denominator_include_pos = 'False')
    # print(out)
    
    
    # iqa_metric = pyiqa.create_metric('musiq', as_loss=True).cuda()
