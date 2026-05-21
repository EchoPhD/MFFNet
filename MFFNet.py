import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn
from scipy.io import savemat
from torchvision import models
from einops import rearrange
import numbers
from torchvision import transforms
from scipy.io import loadmat, savemat
from DWT_IDWT.DWT_IDWT_layer import DWT_1D, DWT_2D, IDWT_1D, IDWT_2D

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
#torch_ver = torch.__version__[:3]

import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

import settings
#from hamburger import ConvBNReLU, get_hamburger
from hamburger import get_hamburger

from scipy.sparse.linalg import cg
import scipy.sparse as sp
''' ================================================= '''
''' =================== cat + cnn =================== '''
''' ================================================= '''

'''------------------------------------------------
conv1x1函数： 定义1×1卷积层
输入：输入通道数、输出通道数
输出：设置好的卷积层
------------------------------------------------'''
def conv1x1(in_channels, out_channels, stride=1):
    #return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

'''------------------------------------------------
conv3x3函数： 定义3×3卷积层
输入：输入通道数、输出通道数
输出：设置好的卷积层
------------------------------------------------'''
def conv3x3(in_channels, out_channels, stride=1):
    #return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

'''------------------------------------------------
ResBlock函数： 定义残差块网络
输入：输入通道数、输出通道数
输出：两层3×3卷积、ReLU激活的残差块网络
------------------------------------------------'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out
    
    
'''========================================================================
Feature_extraction模块： 浅特征提取，卷积+残差+卷积 用于通道调整
输入：原始尺寸的HSI/pan
输出：256通道
========================================================================'''
class Feature_extraction(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(Feature_extraction, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        
        # 批归一化和层归一化
        self.OutBN = nn.BatchNorm2d(num_features=n_feats)  
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        #x = x + x1
        
        #x = self.OutBN(x)
        #x = F.relu(x)
        return x

'''========================================================================
Feature_adjustment模块：特征调整模块，卷积+残差+卷积
输入：原始尺寸的HSI
输出：HSI特征图
========================================================================'''
class Feature_adjustment(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(Feature_adjustment, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        #x = F.relu(self.conv_tail(x))
        x = self.conv_tail(x)
        #x = x + x1
        return x


'''===========================================================================
晶格模块中PAN的空间注意力模块(a part of CBAM)：
输入：通道调整后的PAN
输出：空间增强后的权重A
===========================================================================''' 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)      #self.sigmoid(x) * x

'''===========================================================================
晶格模块中HSI的通道注意力模块(a part of CBAM)：
输入：通道调整后的HSI
输出：通道增强后权重B
===========================================================================''' 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享权重的MLP
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)        #self.sigmoid(out) * x

'''========================================================================
晶格模块： 用于替换原始残差第一阶段融合HSI与PAN
输入：原始尺寸的HSI/pan
输出：256通道融合结果
========================================================================'''
class LatticeBlock(nn.Module):
    def __init__(self, nFeat):
        super(LatticeBlock, self).__init__()
        ''' 此版本是两阶段的晶格结构，第一阶段是混合的，PAN和HSI两个分支混合，第二阶段是在第一阶段最后线性层输出的结果上进行单图处理 '''
        self.n_feats = nFeat    #PAN和HSI输入特征层数

        ''' ================================ 第 1 阶段 ===================================== '''
        #PAN支路的RBs块
        self.conv_head = conv3x3(1, self.n_feats)
        self.conv_block0 = nn.ModuleList()
        for i in range(2):
            self.conv_block0.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats, res_scale=1))
        self.pan_conv_tail = conv3x3(self.n_feats, self.n_feats)

        '''注：这两块的输出结果是权重'''
        self.PAN_A_i_1 = SpatialAttention(kernel_size=7)
        self.HSI_B_i_1 = ChannelAttention(nFeat, ratio=8)

        #HSI支路的RBs块
        self.conv_block1 = nn.ModuleList()
        for i in range(2):
            self.conv_block1.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats, res_scale=1))
        self.hsi_conv_tail = conv3x3(self.n_feats, self.n_feats)

        '''注：这两块的输出结果是权重'''
        self.PAN_A_i = SpatialAttention(kernel_size=7)
        self.HSI_B_i = ChannelAttention(nFeat, ratio=8)
        
        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)
        
    #def forward(self, x):
    def forward(self, pan, hsi):
        
        ''' ==================================== 第 1 阶段 ========================================= '''
        ''' analyse unit '''
        pan = F.relu(self.conv_head(pan))   #pan图调整通道到256
        pan_feature_i_1 = pan
        #pan图经过残差块
        for i in range(2):
            pan_feature_i_1 = self.conv_block0[i](pan_feature_i_1)
        pan_feature_i_1 = self.pan_conv_tail(pan_feature_i_1)
        
        pan_feature_A_i_1 = self.PAN_A_i_1(pan_feature_i_1) #第一个晶格结构PAN图权重A_i-1
        hsi_feature_B_i_1 = self.HSI_B_i_1(hsi) #第一个晶格结构HSI图权重B_i-1        
        ##开始相加
        hsi_next = hsi + pan_feature_A_i_1*pan_feature_i_1  #HSI + PAN权重A_i-1 * PAN
        pan_next = pan_feature_i_1 + hsi_feature_B_i_1* hsi  #PAN + HSI权重B_i-1 * HSI
        

        ''' synthes unit '''
        hsi_feature_i = hsi_next
        #HSI图经过残差块
        for i in range(2):
            hsi_feature_i = self.conv_block1[i](hsi_feature_i)
        hsi_feature_i = self.hsi_conv_tail(hsi_feature_i)
        
        hsi_feature_B_i = self.HSI_B_i(hsi_feature_i)     #第二个晶格结构HSI图权重B_i   
        pan_feature_A_i = self.PAN_A_i(pan_next)  #第二个晶格结构PAN图权重A_i
        
        hsi_finial = hsi_feature_i + pan_feature_A_i*pan_next
        pan_finial = pan_next + hsi_feature_B_i*hsi_feature_i      

        out = torch.cat((hsi_finial, pan_finial), 1)
        out = self.compress(out)
                      
        return out

'''========================================================================
DWT_data函数： 小波变换/逆变换
输入：[batch_size, channels, image_size,image_size]
输出：data_ave, data_hor, data_ver, data_dia （低频均值分量、水平高频lh、垂直高频hl、对角高频hh）
========================================================================'''
def DWT_data(self, data, I_NO):
    ''' I_NO=0, 则进行小波变换，否则进行小波逆变换'''
    if not I_NO:
        # 2位小波变换 一种比较简单的小波 haar(哈尔小波）
        dwt = DWT_2D("haar")
        #xll, xlh, xhl, xhh = dwt(data)
        final_result = dwt(data)
    else:
        iwt = IDWT_2D("haar")
        final_result = iwt(data[0],data[1],data[2],data[3])

    #####以下内容为保存小波变换的结果为mat文件
    # filepath = "./temp_data/"
    ####tensor转numpy
    # xll = xll.cpu().detach().numpy()
    # xlh = xlh.cpu().detach().numpy()
    # xhl = xhl.cpu().detach().numpy()
    # xhh = xhh.cpu().detach().numpy()
    # dwt_data = dwt_data.cpu().detach().numpy()
    # savemat("xll.mat", {'xll':xll})
    # savemat("xlh.mat", {'xlh':xlh})
    # savemat("xhl.mat", {'xhl':xhl})
    # savemat("xhh.mat", {'xhh':xhh})
    # savemat("dwt_data.mat", {'dwt_data': dwt_data})
    # return [xll,xlh,xhl,xhh]
    return final_result



'''================================================
attention模块：self attention或者cross attention模块。 
输入：
输出：
================================================'''
class Attention(nn.Module):
    ''' Scaled Dot-Product Attention 缩放点积注意'''
    def __init__(self, channels,num_heads):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm_q_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层Q支路LN归一化层
        self.norm_k_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层K支路LN归一化层
        self.norm_v_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层V支路LN归一化层
        
        '''第一层Q支路中的 1×1 + 3×3 卷积层'''
        self.Q_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))        

        '''第一层K支路中的 1×1 + 3×3 卷积层'''
        self.K_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))
        
        '''第一层V支路中的 1×1 + 3×3 卷积层'''
        self.V_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))   
        
        # 批归一化和层归一化
        self.OutBN = nn.BatchNorm2d(num_features = self.channels)              
        self.project_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)        
    
    def forward(self, v_01, k_01, q_01):
        output_01 = v_01    #加入DHP
        
        ''' 归一化层 '''
        q_01 = self.norm_q_01(q_01)   #第一层Q支路LN归一化层
        k_01 = self.norm_k_01(k_01)   #第一层K支路LN归一化层
        v_01 = self.norm_v_01(v_01)   #第一层V支路LN归一化层        

        ''' 1*1 + 3*3 卷积获取通道间上下文 '''
        q_01 = self.Q_01(q_01)
        k_01 = self.K_01(k_01)
        v_01 = self.V_01(v_01)

        '''重塑K,Q和V...'''                
        b, c, h, w = q_01.size(0), q_01.size(1), q_01.size(2), q_01.size(3)
        #q_01 = q_01.view(b, c, h*w)
        #k_01 = k_01.view(b, c, h*w)
        #v_01 = v_01.view(b, c, h*w)

        q_01 = rearrange(q_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_01 = rearrange(k_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_01 = rearrange(v_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        '''自注意力注意力计算'''
        #output_01 = v_01    
        q_01 = torch.nn.functional.normalize(q_01, dim=-1)
        k_01 = torch.nn.functional.normalize(k_01, dim=-1)
        # 计算注意力      
        attn_01 = (q_01 @ k_01.transpose(-2, -1)) * self.temperature
        #标准化(SoftMax)
        attn_01    = F.softmax(attn_01, dim=-1)
        out = (attn_01 @ v_01)
        out = output_01 + rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        output = self.project_out(out)
        output = F.relu(self.OutBN(output))
        
        return output

'''================================================
Cross_Attention模块：cross attention模块。 
输入：
输出：
================================================'''
class Cross_Attention(nn.Module):
    ''' Scaled Dot-Product Attention 缩放点积注意'''
    def __init__(self, channels,num_heads):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.fliper = transforms.RandomHorizontalFlip(1)        
             
        self.norm_q_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层Q支路LN归一化层
        self.norm_k_01 = nn.BatchNorm2d(num_features = self.channels*2)   #第一层K支路LN归一化层
        self.norm_v_01 = nn.BatchNorm2d(num_features = self.channels*2)   #第一层V支路LN归一化层

        '''第一层Q支路中的 1×1 + 3×3 卷积层'''
        self.Q_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))        

        '''第一层K支路中的 1×1 + 3×3 卷积层'''
        self.K_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))
        
        '''第一层V支路中的 1×1 + 3×3 卷积层'''
        self.V_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))   
        
        # 批归一化和层归一化
        self.OutBN = nn.BatchNorm2d(num_features = self.channels)              
        self.project_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)        
    
    def forward(self, v_01, k_01, q_01):
        output_01 = v_01    
        
        ''' 归一化层 '''
        q_01 = self.norm_q_01(q_01)   #第一层Q支路LN归一化层
        
        kv_01= torch.cat((q_01,k_01), dim=1)  
        k_01, v_01 = kv_01,kv_01
        k_01 = self.norm_k_01(k_01)   #第一层K支路LN归一化层
        v_01 = self.norm_v_01(v_01)   #第一层V支路LN归一化层        

        ''' 1*1 + 3*3 卷积获取通道间上下文 '''      
        q_01 = self.Q_01(q_01)        
        k_01 = self.K_01(k_01)
        v_01 = self.V_01(v_01)

        '''重塑K,Q和V...'''                
        b, c, h, w = q_01.size(0), q_01.size(1), q_01.size(2), q_01.size(3)
        q_01 = rearrange(q_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_01 = rearrange(k_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_01 = rearrange(v_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        '''自注意力注意力计算'''
        #output_01 = v_01    
        q_01 = torch.nn.functional.normalize(q_01, dim=-1)
        k_01 = torch.nn.functional.normalize(k_01, dim=-1)
        # 计算注意力      
        attn_01 = (q_01 @ k_01.transpose(-2, -1)) * self.temperature
        #标准化(SoftMax)
        attn_01    = F.softmax(attn_01, dim=-1)
        out = (attn_01 @ v_01)
        out = output_01 + rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        output = self.project_out(out)
        #output = F.relu(self.OutBN(output))
        output = self.OutBN(output)
        
        return output

class ESSAttn(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        #b, N, C = x.shape
        b, C, H, W = x.shape
        #x = x.view(b, H*W, C)
        N = H*W
        x = rearrange(x, 'b C H W -> b (H W) C')

        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        attn = t1 + t2
        attn = self.ln(attn)
        # print("attn.shape:---------------->", attn.shape)
        #attn = x.view(b, C, H, W)
        attn = attn.permute(0, 2, 1).contiguous().view(b, C, H, W)
        # attn = rearrange(attn, 'b (H W) C -> b C H W', H=H, W=W)
        # print("attn.shape:---------------->", attn.shape)
        return attn

    def is_same_matrix(self, m1, m2):
        rows, cols = len(m1), len(m1[0])
        for i in range(rows):
            for j in range(cols):
                if m1[i][j] != m2[i][j]:
                    return False
        return True

''' 曲率通道注意力_01
#@torch.no_grad()
def curvature_spectral_attention(input_tensor):
    pass
'''

''' _02 
def curvature_spatial_attention(x):
    pass
'''


''' =====名为 CurvMap 的自定义 PyTorch 模块，用于计算输入图像的曲率地图===== '''
class CurvMap(nn.Module):
    def __init__(self, scale=1):
        super(CurvMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        # Initialize an empty tensor to store the curvature maps for each channel
        curvature_maps = []

        for channel in range(img.size(1)):  # Iterate over channels
            channel_img = img[:, channel:channel + 1, :, :]  # Extract the current channel

            # Perform the same operations as before for each channel
            channel_img = channel_img / self.scale
            #channel_img = TF.rgb_to_grayscale(channel_img)
            channel_img_pad = F.pad(channel_img, pad=(1, 1, 1, 1), mode='reflect')

            N, C, H, W = img.shape
            gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
            gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
            gradXX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
            gradXY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
            gradYY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

            # 计算输入图像的一阶和二阶导数
            gradx = (channel_img[..., 1:, :] - channel_img[..., :-1, :]).abs()
            grady = (channel_img[..., 1:] - channel_img[..., :-1]).abs()
            gradxx = (channel_img_pad[..., 2:, 1:-1] + channel_img_pad[..., :-2, 1:-1] - 2 * channel_img_pad[..., 1:-1, 1:-1]).abs()
            gradyy = (channel_img_pad[..., 1:-1, 2:] + channel_img_pad[..., 1:-1, :-2] - 2 * channel_img_pad[..., 1:-1, 1:-1]).abs()
            gradxy = (channel_img_pad[..., 2:, 2:] + channel_img_pad[..., 1:-1, 1:-1] - channel_img_pad[..., 2:, 1:-1] - channel_img_pad[..., 1:-1, 2:]).abs()

            # 计算横向梯度
            gradX[..., :-1, :] += gradx
            gradX[..., 1:, :] += gradx
            gradX[..., 1:-1, :] /= 2

            # 计算纵向梯度
            gradY[..., :-1] += grady
            gradY[..., 1:] += grady
            gradY[..., 1:-1] /= 2
            
            # 将计算的二阶导数存储到相应的张量中
            gradXX = gradxx
            gradYY = gradyy
            gradXY = gradxy

            curvature = (gradYY * (1 + torch.square(gradX)) - 2 * gradXY * gradX * gradY + gradXX * (1 + torch.square(gradY))) / \
                        torch.sqrt(torch.pow((torch.square(gradX) + torch.square(gradY) + 1), 3))

            #print("curvature.shape:------------------>",curvature.shape)
            curvature_maps.append(curvature)

        # 将曲率地图沿通道维度堆叠
        #curvature_maps = torch.stack(curvature_maps, dim=1)
        curvature_maps = torch.cat(curvature_maps, dim=1)
        #print("curvature_maps.shape:--------------->", curvature_maps.shape)

        return curvature_maps

'''===========================================================================
curvature Spatial Attention：
输入：通道调整后的curvature map
输出：空间增强后的权重A
===========================================================================''' 
class curvature_spatial_attention(nn.Module):
    #def __init__(self, kernel_size=7):
    def __init__(self,):
        super(curvature_spatial_attention, self).__init__()
        #assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #padding = 3 if kernel_size == 7 else 1
        #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        #self.conv1 = conv1x1(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #x = torch.cat([avg_out, max_out], dim=1)
        #x = self.conv1(x)
        x = max_out + avg_out
        return self.sigmoid(x)      #self.sigmoid(x) * x

'''===========================================================================
curvature channel Attention：
输入：curvature map
输出：通道增强后权重B
===========================================================================''' 
class curvature_channel_attention(nn.Module):
    #def __init__(self, in_planes, ratio=16):
    def __init__(self,):
        super(curvature_channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享权重的MLP
        #self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        #self.relu1 = nn.ReLU()
        #self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        #max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = avg_out + max_out
        return self.sigmoid(out)        #self.sigmoid(out) * x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0), nn.LeakyReLU(),nn.Conv2d(in_channels, out_channels, 1, 1, 0), nn.LeakyReLU())

    def forward(self, x1):
        return self.up1(x1)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(in_channels, out_channels, 1, 1, 0),nn.LeakyReLU())
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 2, 2, 0), nn.LeakyReLU(), nn.Conv2d(in_channels, out_channels, 1, 1, 0), nn.LeakyReLU())

    def forward(self, x):
        return self.conv(x)


'''================================================
DWTransformer模块：总启动程序 
输入：
输出：
================================================'''
class MDTransformerpre(nn.Module):
    def __init__(self, config):
        super(MDTransformerpre, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.num_head      = config["N_modules"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"] #光谱通道数
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]     #尺寸缩放比例

        self.num_res_blocks = [1, 1] #Feature_extraction模块与Feature_adjustment模块中所用的残差块数量
        self.unet_res_blocks = [1, 1, 1, 1, 1, 1]  # U-net中所用的残差块数量
        self.res_scale      = 1 #残差缩放，默认为1 
        self.feature_chanels = [128, 256, 512, 256, 256]

        ''' 矩阵分解的三个超参数 '''
        #分解的尺度 MD_S、基础矩阵的维度 MD_D 和基础矩阵的秩 MD_R
        self.MD_S = [1, 1, 1, 1, 1]
        self.MD_D = [128, 256, 512, 256, 256]
        self.MD_R = [18, 32, 64, 32, 32]
        
        ''' LR-HSI和PAN的特征通道调整(Feature_extraction模块) '''      
        #self.out_feature_extraction = Feature_extraction(self.in_channels+1, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)
        self.up_lrhsi_feature_extraction = Feature_extraction(self.in_channels, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)
        self.pan_feature_extraction = Feature_extraction(1, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)

        ''' =========== 使用之前定义的 get_hamburger 函数构建一个 Hamburger 模块，并初始化 self.hamburger =========== '''
        ''' =========== U-net中依次使用的矩阵分解块定义 =========== '''
        # 00
        #C = settings.CHANNELS
        Hamburger_00 = get_hamburger(settings.VERSION)
        #self.hamburger = Hamburger(C, settings)
        self.hamburger_00 = Hamburger_00(self.feature_chanels[0], settings, self.MD_S[0], self.MD_D[0], self.MD_R[0])

        # 01
        Hamburger_01 = get_hamburger(settings.VERSION)
        self.hamburger_01 = Hamburger_01(self.feature_chanels[1], settings, self.MD_S[1], self.MD_D[1], self.MD_R[1])

        # 02
        Hamburger_02 = get_hamburger(settings.VERSION)
        self.hamburger_02 = Hamburger_02(self.feature_chanels[2], settings, self.MD_S[2], self.MD_D[2], self.MD_R[2])

        # 03
        Hamburger_03 = get_hamburger(settings.VERSION)
        self.hamburger_03 = Hamburger_03(self.feature_chanels[3], settings, self.MD_S[3], self.MD_D[3], self.MD_R[3])


        ''' ====================== PAN支路 上的残差块 一共四个 ============================ '''
        # 00
        # self.pan_res_block00 = nn.ModuleList()
        # for i in range(self.unet_res_blocks[0]):
        #     self.pan_res_block00.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        # self.pan_res_block00_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        # 01
        self.pan_res_block01 = nn.ModuleList()
        for i in range(self.unet_res_blocks[1]):
            self.pan_res_block01.append(ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.pan_res_block01_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 02
        self.pan_res_block02 = nn.ModuleList()
        for i in range(self.unet_res_blocks[2]):
            self.pan_res_block02.append(ResBlock(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], res_scale=1))
        self.pan_res_block02_conv = conv3x3(self.feature_chanels[2], self.feature_chanels[2])

        # 03
        self.pan_res_block03 = nn.ModuleList()
        for i in range(self.unet_res_blocks[3]):
            self.pan_res_block03.append(ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.pan_res_block03_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])

        ''' ===================== HS支路 上的残差块 一共四个 ======================='''
        # 00
        # self.hs_res_block00 = nn.ModuleList()
        # for i in range(self.unet_res_blocks[0]):
        #     self.hs_res_block00.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        # self.hs_res_block00_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        # 01
        self.hs_res_block01 = nn.ModuleList()
        for i in range(self.unet_res_blocks[1]):
            self.hs_res_block01.append(ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.hs_res_block01_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 02
        self.hs_res_block02 = nn.ModuleList()
        for i in range(self.unet_res_blocks[2]):
            self.hs_res_block02.append(ResBlock(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], res_scale=1))
        self.hs_res_block02_conv = conv3x3(self.feature_chanels[2], self.feature_chanels[2])

        # 03
        self.hs_res_block03 = nn.ModuleList()
        for i in range(self.unet_res_blocks[3]):
            self.hs_res_block03.append(ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.hs_res_block03_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])


        ''' ======================= 曲率注意力计算 curvature attention ====================== '''
        self.img_curv = CurvMap()
        #PAN之路上的四个曲率空间注意力
        self.curvature_SA_00 = curvature_spatial_attention()
        self.curvature_SA_01 = curvature_spatial_attention()
        self.curvature_SA_02 = curvature_spatial_attention()
        self.curvature_SA_03 = curvature_spatial_attention()
        #HS支路上的四个曲率通道注意力
        self.curvature_CA_00 = curvature_channel_attention()
        self.curvature_CA_01 = curvature_channel_attention()
        self.curvature_CA_02 = curvature_channel_attention()
        self.curvature_CA_03 = curvature_channel_attention()

        ''' ======================== 降采样操作 ========================== '''
        # 邓哥论文中两个支路共享了上采样和下采样参数
        # PAN支路两个降采样
        self.pan_down00 = Down(self.feature_chanels[0], self.feature_chanels[1])
        self.pan_down01 = Down(self.feature_chanels[1], self.feature_chanels[2])
        # HS支路两个降采样
        self.hs_down00 = Down(self.feature_chanels[0], self.feature_chanels[1])
        self.hs_down01 = Down(self.feature_chanels[1], self.feature_chanels[2])


        ''' ======================== 上采样操作 ========================== '''
        # PAN支路两个上采样
        self.pan_up00 = Up(self.feature_chanels[2], self.feature_chanels[1])
        self.pan_up01 = Up(self.feature_chanels[1], self.feature_chanels[0])
        # HS支路两个上采样
        self.hs_up00 = Up(self.feature_chanels[2], self.feature_chanels[1])
        self.hs_up01 = Up(self.feature_chanels[1], self.feature_chanels[0])

        ''' ======================== 最后融合模块操作设置 ========================== '''
        # 进入融合块时的残差 cat
        self.fusion_res_block = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.fusion_res_block.append(ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.fusion_res_block_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])
        
        ''' # 进入融合块时的残差 no cat
        self.fusion_res_block_pan = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.fusion_res_block_pan.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        self.fusion_res_block_pan_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        self.fusion_res_block_hs = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.fusion_res_block_hs.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        self.fusion_res_block_hs_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])
        '''
        
        # 空间注意力支路上的残差
        self.sa_res_block = nn.ModuleList()
        for i in range(self.unet_res_blocks[5]):
            self.sa_res_block.append(ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.sa_res_block_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 通道注意力支路上的残差
        self.ca_res_block = nn.ModuleList()
        for i in range(self.unet_res_blocks[5]):
            self.ca_res_block.append(ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.ca_res_block_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 曲率注意力计算 curvature attention
        #空间支路上的曲率空间注意力
        self.curvature_SA = curvature_spatial_attention()
        #通道支路上的曲率通道注意力
        self.curvature_CA = curvature_channel_attention()

        # 矩阵分解
        Hamburger_final = get_hamburger(settings.VERSION)
        self.Hamburger_final = Hamburger_final(self.feature_chanels[4], settings, self.MD_S[4], self.MD_D[4], self.MD_R[4])
        #no cat
        #self.Hamburger_final = Hamburger_final(self.feature_chanels[0], settings, self.MD_S[0], self.MD_D[0], self.MD_R[0])

        # 相加后通过最后一层卷积调整通道，输出最终的融合结果
        self.compress = nn.Conv2d(self.feature_chanels[1], self.out_channels, kernel_size=1, padding=0, bias=True)
        # no cat
        #self.compress = nn.Conv2d(self.feature_chanels[0], self.out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, LR_HSI, PAN):
        #调整PAN尺寸[batch_size, 通道数，HR_size, HR_size]
        PAN = PAN.unsqueeze(dim=1)
        
        UP_LR_HSI = F.interpolate(LR_HSI, scale_factor=(self.factor,self.factor),mode ='bicubic')   #上采样的LR-HSI，用于晶格结构
        F_UP_LR_HSI = self.up_lrhsi_feature_extraction(UP_LR_HSI)
        
        F_PAN = self.pan_feature_extraction(PAN)    # [B, C, H, W]

        ''' ====================================== 第一尺度/编码器 ================================== '''
        F_UP_LR_HSI_00 = F_UP_LR_HSI
        F_PAN_00 = F_PAN

        ''' 曲率空间/通道注意力计算 
        F_UP_LR_HSI_curvature_00 = self.img_curv(F_UP_LR_HSI_00)
        F_PAN_curvature_00 = self.img_curv(F_PAN_00)
        F_UP_LR_HSI_curvature_00 = self.curvature_CA_00(F_UP_LR_HSI_curvature_00)
        F_PAN_curvature_00 = self.curvature_SA_00(F_PAN_curvature_00)
        F_UP_LR_HSI = F_UP_LR_HSI_curvature_00 * F_UP_LR_HSI    #加权计算
        F_PAN = F_PAN_curvature_00 * F_PAN
        '''
        
        # 矩阵分解混合注意力计算
        #F_MD_00 = self.hamburger_00(F_PAN_00, F_UP_LR_HSI_00)   #(通道，空间)
        F_MD_00 = self.hamburger_00(F_UP_LR_HSI_00, F_PAN_00)   #(通道，空间)
        F_UP_LR_HSI_MD_00 = F_UP_LR_HSI + F_MD_00["x_01"]
        F_PAN_MD_00 = F_PAN + F_MD_00["x_02"]
        # 留出跳转连接变量
        F_UP_LR_HSI_MD_skip_00 = F_UP_LR_HSI_MD_00
        F_PAN_MD_skip_00 = F_PAN_MD_00
        #print("F_UP_LR_HSI_MD_skip_00.shape:--------------------->",F_UP_LR_HSI_MD_skip_00.shape)   # [8, 128, 160, 160]
        #print("F_PAN_MD_skip_00.shape:--------------------->",F_PAN_MD_skip_00.shape)   # [8, 128, 160, 160]
        #下采样
        F_UP_LR_HSI_00 = self.hs_down00(F_UP_LR_HSI_MD_00)  # [B, 2C, H/2, W/2]
        F_PAN_00 = self.pan_down00(F_PAN_MD_00)


        ''' =========================== 第二尺度/编码器 =============================== '''
        # 经过残差
        for i in range(self.unet_res_blocks[1]):
            F_UP_LR_HSI_00 = self.hs_res_block01[i](F_UP_LR_HSI_00)
        F_UP_LR_HSI_00 = self.hs_res_block01_conv(F_UP_LR_HSI_00)

        for i in range(self.unet_res_blocks[1]):
            F_PAN_00 = self.pan_res_block01[i](F_PAN_00)
        F_PAN_00 = self.pan_res_block01_conv(F_PAN_00)

        F_UP_LR_HSI_01 = F_UP_LR_HSI_00
        F_PAN_01 = F_PAN_00

        ''' 曲率空间/通道注意力计算 
        F_UP_LR_HSI_curvature_01 = self.img_curv(F_UP_LR_HSI_01)
        F_PAN_curvature_01 = self.img_curv(F_PAN_01)
        F_UP_LR_HSI_curvature_01 = self.curvature_CA_01(F_UP_LR_HSI_curvature_01)
        F_PAN_curvature_01 = self.curvature_SA_01(F_PAN_curvature_01)
        F_UP_LR_HSI_00 = F_UP_LR_HSI_curvature_01 * F_UP_LR_HSI_00    #加权计算
        F_PAN_00 = F_PAN_curvature_01 * F_PAN_00
        '''
        
        # 矩阵分解混合注意力计算
        #F_MD_01 = self.hamburger_01(F_PAN_01, F_UP_LR_HSI_01)   #(通道，空间)
        F_MD_01 = self.hamburger_01(F_UP_LR_HSI_01, F_PAN_01)   #(通道，空间)
        F_UP_LR_HSI_MD_01 = F_UP_LR_HSI_00 + F_MD_01["x_01"]
        F_PAN_MD_01 = F_PAN_00 + F_MD_01["x_02"]
        # 留出跳转连接变量
        F_UP_LR_HSI_MD_skip_01 = F_UP_LR_HSI_MD_01
        F_PAN_MD_skip_01 = F_PAN_MD_01
        #print("F_UP_LR_HSI_MD_skip_01.shape:--------------------->",F_UP_LR_HSI_MD_skip_01.shape)   # [8, 256, 80, 80]
        #print("F_PAN_MD_skip_01.shape:--------------------->",F_PAN_MD_skip_01.shape)   # [8, 256, 80, 80]
        #下采样
        F_UP_LR_HSI_01 = self.hs_down01(F_UP_LR_HSI_MD_01)  # [B, 4C, H/4, W/4]
        F_PAN_01 = self.pan_down01(F_PAN_MD_01)

        ''' ============================== 第二尺度/解码器 =============================== '''
        # 经过残差
        for i in range(self.unet_res_blocks[2]):
            F_UP_LR_HSI_01 = self.hs_res_block02[i](F_UP_LR_HSI_01)
        F_UP_LR_HSI_01 = self.hs_res_block02_conv(F_UP_LR_HSI_01)
        for i in range(self.unet_res_blocks[2]):
            F_PAN_01 = self.pan_res_block02[i](F_PAN_01)
        F_PAN_01 = self.pan_res_block02_conv(F_PAN_01)

        F_UP_LR_HSI_02 = F_UP_LR_HSI_01
        F_PAN_02 = F_PAN_01

        ''' 曲率空间/通道注意力计算 
        F_UP_LR_HSI_curvature_02 = self.img_curv(F_UP_LR_HSI_02)
        F_PAN_curvature_02 = self.img_curv(F_PAN_02)
        F_UP_LR_HSI_curvature_02 = self.curvature_CA_02(F_UP_LR_HSI_curvature_02)
        F_PAN_curvature_02 = self.curvature_SA_02(F_PAN_curvature_02)
        F_UP_LR_HSI_01 = F_UP_LR_HSI_curvature_02 * F_UP_LR_HSI_01    #加权计算
        F_PAN_01 = F_PAN_curvature_02 * F_PAN_01
        '''
        
        # 矩阵分解混合注意力计算
        #F_MD_02 = self.hamburger_02(F_PAN_02, F_UP_LR_HSI_02)   #(通道，空间)
        F_MD_02 = self.hamburger_02(F_UP_LR_HSI_02, F_PAN_02)   #(通道，空间)
        F_UP_LR_HSI_MD_02 = F_UP_LR_HSI_01 + F_MD_02["x_01"]
        F_PAN_MD_02 = F_PAN_01 + F_MD_02["x_02"]
        #上采样
        F_UP_LR_HSI_MD_02 = self.hs_up00(F_UP_LR_HSI_MD_02)  # [B, 2C, H/2, W/2]
        F_PAN_MD_02 = self.pan_up00(F_PAN_MD_02)        
        #print("F_UP_LR_HSI_MD_02.shape:--------------------->",F_UP_LR_HSI_MD_02.shape) # [8, 256, 80, 80]
        #print("F_PAN_MD_02.shape:--------------------->",F_PAN_MD_02.shape) # [8, 256, 80, 80]        
        
        # 加上跳转连接变量
        F_UP_LR_HSI_02 = F_UP_LR_HSI_MD_02 + F_UP_LR_HSI_MD_skip_01
        F_PAN_02 = F_PAN_MD_02 + F_PAN_MD_skip_01

        ''' =============================== 第一尺度/解码器 =================================== '''
        # 经过残差
        for i in range(self.unet_res_blocks[3]):
            F_UP_LR_HSI_02 = self.hs_res_block03[i](F_UP_LR_HSI_02)
        F_UP_LR_HSI_02 = self.hs_res_block03_conv(F_UP_LR_HSI_02)
        for i in range(self.unet_res_blocks[3]):
            F_PAN_02 = self.pan_res_block03[i](F_PAN_02)
        F_PAN_02 = self.pan_res_block03_conv(F_PAN_02)

        F_UP_LR_HSI_03 = F_UP_LR_HSI_02
        F_PAN_03 = F_PAN_02

        ''' 曲率空间/通道注意力计算 
        F_UP_LR_HSI_curvature_03 = self.img_curv(F_UP_LR_HSI_03)
        F_PAN_curvature_03 = self.img_curv(F_PAN_03)
        F_UP_LR_HSI_curvature_03 = self.curvature_CA_03(F_UP_LR_HSI_curvature_03)
        F_PAN_curvature_03 = self.curvature_SA_03(F_PAN_curvature_03)
        F_UP_LR_HSI_02 = F_UP_LR_HSI_curvature_03 * F_UP_LR_HSI_02  # 加权计算
        F_PAN_02 = F_PAN_curvature_03 * F_PAN_02
        '''
        
        # 矩阵分解混合注意力计算
        #F_MD_03 = self.hamburger_03(F_PAN_03, F_UP_LR_HSI_03)  # (通道，空间)
        F_MD_03 = self.hamburger_03(F_UP_LR_HSI_03, F_PAN_03)  # (通道，空间)
        F_UP_LR_HSI_MD_03 = F_UP_LR_HSI_02 + F_MD_03["x_01"]
        F_PAN_MD_03 = F_PAN_02 + F_MD_03["x_02"]

        # 上采样
        F_UP_LR_HSI_MD_03 = self.hs_up01(F_UP_LR_HSI_MD_03)  # [B, C, H, W]
        F_PAN_MD_03 = self.pan_up01(F_PAN_MD_03)        
        #print("F_UP_LR_HSI_MD_03.shape:--------------------->",F_UP_LR_HSI_MD_03.shape) # [8, 128, 160, 160] 
        #print("F_PAN_MD_03.shape:--------------------->",F_PAN_MD_03.shape) # [8, 128, 160, 160] 
        
        # 加上跳转连接变量
        F_UP_LR_HSI_03 = F_UP_LR_HSI_MD_03 + F_UP_LR_HSI_MD_skip_00
        F_PAN_03 = F_PAN_MD_03 + F_PAN_MD_skip_00

        ''' ======================== 最终的融合模块 ====================== '''
        #cat or no cat
        F_fusion = torch.cat((F_PAN_03, F_UP_LR_HSI_03), dim = 1)  #[B, 2C, H, W]

        #初始残差 cat
        for i in range(self.unet_res_blocks[4]):
            F_fusion = self.fusion_res_block[i](F_fusion)
        F_fusion = self.fusion_res_block_conv(F_fusion)
        
        '''# no cat        
        for i in range(self.unet_res_blocks[4]):
            F_PAN_03 = self.fusion_res_block_pan[i](F_PAN_03)
        F_PAN = self.fusion_res_block_pan_conv(F_PAN_03)
        for i in range(self.unet_res_blocks[4]):
            F_UP_LR_HSI_03 = self.fusion_res_block_hs[i](F_UP_LR_HSI_03)
        F_UP_LR_HSI = self.fusion_res_block_hs_conv(F_UP_LR_HSI_03)
        '''
        
        # 曲率空间/通道注意力计算
        F_fusion_ca = F_fusion
        F_fusion_sa = F_fusion       

        #F_fusion_curvature = self.img_curv(F_fusion_ca)
        #F_fusion_curvature_ca = self.curvature_CA(F_fusion_curvature)
        #F_fusion_curvature_sa = self.curvature_SA(F_fusion_curvature)
        #F_fusion_ca = F_fusion_curvature_ca * F_fusion    #加权计算
        #F_fusion_sa = F_fusion_curvature_sa * F_fusion

        # 矩阵分解混合注意力计算
        F_MD = self.Hamburger_final(F_fusion_ca, F_fusion_sa)   #(通道，空间)
        F_UP_LR_HSI_MD = F_fusion_ca + F_MD["x_01"]
        F_PAN_MD = F_fusion_sa + F_MD["x_02"]

        F_fusion_final = F_UP_LR_HSI_MD + F_PAN_MD

        F_out = self.compress(F_fusion_final)

        output = {"pred": F_out}

        return output
    
    
'''================================================
DWTransformer模块：总启动程序 
输入：
输出：
================================================'''
class MDTransformer(nn.Module):
    def __init__(self, config):
        super(MDTransformer, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.num_head      = config["N_modules"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"] #光谱通道数
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]     #尺寸缩放比例

        self.num_res_blocks = [1, 1] #Feature_extraction模块与Feature_adjustment模块中所用的残差块数量
        self.unet_res_blocks = [1, 1, 1, 1, 1, 1]  # U-net中所用的残差块数量
        self.res_scale      = 1 #残差缩放，默认为1 
        self.feature_chanels = [128, 256, 512, 256, 256]

        ''' 矩阵分解的三个超参数 '''
        #分解的尺度 MD_S、基础矩阵的维度 MD_D 和基础矩阵的秩 MD_R
        self.MD_S = [1, 1, 1, 1, 1]
        self.MD_D = [128, 256, 512, 256, 256]
        self.MD_R = [18, 32, 64, 32, 32]
        
        ''' LR-HSI和PAN的特征通道调整(Feature_extraction模块) '''      
        #self.out_feature_extraction = Feature_extraction(self.in_channels+1, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)
        self.up_lrhsi_feature_extraction = Feature_extraction(self.in_channels, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)
        self.pan_feature_extraction = Feature_extraction(1, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)

        ''' =========== 使用之前定义的 get_hamburger 函数构建一个 Hamburger 模块，并初始化 self.hamburger =========== '''
        ''' =========== U-net中依次使用的矩阵分解块定义 =========== '''
        # 00
        #C = settings.CHANNELS
        Hamburger_00 = get_hamburger(settings.VERSION)
        #self.hamburger = Hamburger(C, settings)
        self.hamburger_00 = Hamburger_00(self.feature_chanels[0], settings, self.MD_S[0], self.MD_D[0], self.MD_R[0])

        # 01
        Hamburger_01 = get_hamburger(settings.VERSION)
        self.hamburger_01 = Hamburger_01(self.feature_chanels[1], settings, self.MD_S[1], self.MD_D[1], self.MD_R[1])

        # 02
        Hamburger_02 = get_hamburger(settings.VERSION)
        self.hamburger_02 = Hamburger_02(self.feature_chanels[2], settings, self.MD_S[2], self.MD_D[2], self.MD_R[2])

        # 03
        Hamburger_03 = get_hamburger(settings.VERSION)
        self.hamburger_03 = Hamburger_03(self.feature_chanels[3], settings, self.MD_S[3], self.MD_D[3], self.MD_R[3])

        ''' ====================== PAN支路 上的残差块 一共四个 ============================ '''
        # 00
        # self.pan_res_block00 = nn.ModuleList()
        # for i in range(self.unet_res_blocks[0]):
        #     self.pan_res_block00.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        # self.pan_res_block00_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        # 01
        self.pan_res_block01 = nn.ModuleList()
        for i in range(self.unet_res_blocks[1]):
            self.pan_res_block01.append(ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.pan_res_block01_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 02
        self.pan_res_block02 = nn.ModuleList()
        for i in range(self.unet_res_blocks[2]):
            self.pan_res_block02.append(ResBlock(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], res_scale=1))
        self.pan_res_block02_conv = conv3x3(self.feature_chanels[2], self.feature_chanels[2])

        # 03
        self.pan_res_block03 = nn.ModuleList()
        for i in range(self.unet_res_blocks[3]):
            self.pan_res_block03.append(ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.pan_res_block03_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])

        ''' ===================== HS支路 上的残差块 一共四个 ======================='''
        # 00
        # self.hs_res_block00 = nn.ModuleList()
        # for i in range(self.unet_res_blocks[0]):
        #     self.hs_res_block00.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        # self.hs_res_block00_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        # 01
        self.hs_res_block01 = nn.ModuleList()
        for i in range(self.unet_res_blocks[1]):
            self.hs_res_block01.append(ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.hs_res_block01_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 02
        self.hs_res_block02 = nn.ModuleList()
        for i in range(self.unet_res_blocks[2]):
            self.hs_res_block02.append(ResBlock(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], res_scale=1))
        self.hs_res_block02_conv = conv3x3(self.feature_chanels[2], self.feature_chanels[2])

        # 03
        self.hs_res_block03 = nn.ModuleList()
        for i in range(self.unet_res_blocks[3]):
            self.hs_res_block03.append(ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.hs_res_block03_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])

        ''' ======================= 曲率注意力计算 curvature attention ====================== '''
        self.img_curv = CurvMap()
        #PAN之路上的四个曲率空间注意力
        self.curvature_SA_00 = curvature_spatial_attention()
        self.curvature_SA_01 = curvature_spatial_attention()
        self.curvature_SA_02 = curvature_spatial_attention()
        self.curvature_SA_03 = curvature_spatial_attention()
        #HS支路上的四个曲率通道注意力
        self.curvature_CA_00 = curvature_channel_attention()
        self.curvature_CA_01 = curvature_channel_attention()
        self.curvature_CA_02 = curvature_channel_attention()
        self.curvature_CA_03 = curvature_channel_attention()

        ''' ======================== 降采样操作 ========================== '''
        # 邓哥论文中两个支路共享了上采样和下采样参数
        # PAN支路两个降采样
        self.pan_down00 = Down(self.feature_chanels[0], self.feature_chanels[1])
        self.pan_down01 = Down(self.feature_chanels[1], self.feature_chanels[2])
        # HS支路两个降采样
        self.hs_down00 = Down(self.feature_chanels[0], self.feature_chanels[1])
        self.hs_down01 = Down(self.feature_chanels[1], self.feature_chanels[2])

        ''' ======================== 上采样操作 ========================== '''
        # PAN支路两个上采样
        self.pan_up00 = Up(self.feature_chanels[2], self.feature_chanels[1])
        self.pan_up01 = Up(self.feature_chanels[1], self.feature_chanels[0])
        # HS支路两个上采样
        self.hs_up00 = Up(self.feature_chanels[2], self.feature_chanels[1])
        self.hs_up01 = Up(self.feature_chanels[1], self.feature_chanels[0])

        ''' ======================== 最后融合模块操作设置 ========================== '''
        ''' # 进入融合块时的残差 cat'''
        self.fusion_res_block = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.fusion_res_block.append(ResBlock(in_channels=self.feature_chanels[4], out_channels=self.feature_chanels[4], res_scale=1))
        self.fusion_res_block_conv = conv3x3(self.feature_chanels[4], self.feature_chanels[4])
        

        ''' # 进入融合块时的残差 no cat
        self.fusion_res_block_pan = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.fusion_res_block_pan.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        self.fusion_res_block_pan_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        self.fusion_res_block_hs = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.fusion_res_block_hs.append(ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        self.fusion_res_block_hs_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])
        '''
        
        # 空间注意力支路上的残差
        self.sa_res_block = nn.ModuleList()
        for i in range(self.unet_res_blocks[5]):
            self.sa_res_block.append(ResBlock(in_channels=self.feature_chanels[4], out_channels=self.feature_chanels[4], res_scale=1))
        self.sa_res_block_conv = conv3x3(self.feature_chanels[4], self.feature_chanels[4])

        # 通道注意力支路上的残差
        self.ca_res_block = nn.ModuleList()
        for i in range(self.unet_res_blocks[5]):
            self.ca_res_block.append(ResBlock(in_channels=self.feature_chanels[4], out_channels=self.feature_chanels[4], res_scale=1))
        self.ca_res_block_conv = conv3x3(self.feature_chanels[4], self.feature_chanels[4])

        # 曲率注意力计算 curvature attention
        #空间支路上的曲率空间注意力
        self.curvature_SA = curvature_spatial_attention()
        #通道支路上的曲率通道注意力
        self.curvature_CA = curvature_channel_attention()

        # 矩阵分解
        Hamburger_final = get_hamburger(settings.VERSION)
        self.Hamburger_final = Hamburger_final(self.feature_chanels[4], settings, self.MD_S[4], self.MD_D[4], self.MD_R[4])

        # 相加后通过最后一层卷积调整通道，输出最终的融合结果
        self.compress = nn.Conv2d(self.feature_chanels[4], self.out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, LR_HSI, PAN):
        #调整PAN尺寸[batch_size, 通道数，HR_size, HR_size]
        PAN = PAN.unsqueeze(dim=1)
        ''' one step: cat + CNN(RBs) '''
        UP_LR_HSI = F.interpolate(LR_HSI, scale_factor=(self.factor,self.factor),mode ='bicubic')   #上采样的LR-HSI，用于晶格结构
        #out = torch.cat((PAN, UP_LR_HSI), dim = 1)
        #F_one_fusion = self.out_feature_extraction(out)
        F_UP_LR_HSI = self.up_lrhsi_feature_extraction(UP_LR_HSI)
        F_PAN = self.pan_feature_extraction(PAN)    # [B, C, H, W]

        ''' ====================================== 第一尺度/编码器 ================================== '''
        F_UP_LR_HSI_00 = F_UP_LR_HSI
        F_PAN_00 = F_PAN

        # 曲率空间/通道注意力计算
        F_UP_LR_HSI_curvature_00 = self.img_curv(F_UP_LR_HSI_00)
        F_PAN_curvature_00 = self.img_curv(F_PAN_00)
        F_UP_LR_HSI_curvature_00 = self.curvature_CA_00(F_UP_LR_HSI_curvature_00)
        F_PAN_curvature_00 = self.curvature_SA_00(F_PAN_curvature_00)
        F_UP_LR_HSI = F_UP_LR_HSI_curvature_00 * F_UP_LR_HSI    #加权计算
        F_PAN = F_PAN_curvature_00 * F_PAN
        # 矩阵分解混合注意力计算
        #F_MD_00 = self.hamburger_00(F_PAN_00, F_UP_LR_HSI_00)   #(通道，空间)
        F_MD_00 = self.hamburger_00(F_UP_LR_HSI_00, F_PAN_00)   #(通道，空间)
        F_UP_LR_HSI_MD_00 = F_UP_LR_HSI + F_MD_00["x_01"]
        F_PAN_MD_00 = F_PAN + F_MD_00["x_02"]
        # 留出跳转连接变量
        F_UP_LR_HSI_MD_skip_00 = F_UP_LR_HSI_MD_00
        F_PAN_MD_skip_00 = F_PAN_MD_00
        #print("F_UP_LR_HSI_MD_skip_00.shape:--------------------->",F_UP_LR_HSI_MD_skip_00.shape)   # [8, 128, 160, 160]
        #print("F_PAN_MD_skip_00.shape:--------------------->",F_PAN_MD_skip_00.shape)   # [8, 128, 160, 160]
        #下采样
        F_UP_LR_HSI_00 = self.hs_down00(F_UP_LR_HSI_MD_00)  # [B, 2C, H/2, W/2]
        F_PAN_00 = self.pan_down00(F_PAN_MD_00)


        ''' =========================== 第二尺度/编码器 =============================== '''
        # 经过残差
        for i in range(self.unet_res_blocks[1]):
            F_UP_LR_HSI_00 = self.hs_res_block01[i](F_UP_LR_HSI_00)
        F_UP_LR_HSI_00 = self.hs_res_block01_conv(F_UP_LR_HSI_00)

        for i in range(self.unet_res_blocks[1]):
            F_PAN_00 = self.pan_res_block01[i](F_PAN_00)
        F_PAN_00 = self.pan_res_block01_conv(F_PAN_00)

        F_UP_LR_HSI_01 = F_UP_LR_HSI_00
        F_PAN_01 = F_PAN_00

        # 曲率空间/通道注意力计算
        F_UP_LR_HSI_curvature_01 = self.img_curv(F_UP_LR_HSI_01)
        F_PAN_curvature_01 = self.img_curv(F_PAN_01)
        F_UP_LR_HSI_curvature_01 = self.curvature_CA_01(F_UP_LR_HSI_curvature_01)
        F_PAN_curvature_01 = self.curvature_SA_01(F_PAN_curvature_01)
        F_UP_LR_HSI_00 = F_UP_LR_HSI_curvature_01 * F_UP_LR_HSI_00    #加权计算
        F_PAN_00 = F_PAN_curvature_01 * F_PAN_00
        
        # 矩阵分解混合注意力计算
        # F_MD_01 = self.hamburger_01(F_PAN_01, F_UP_LR_HSI_01)   #(通道，空间)
        F_MD_01 = self.hamburger_01(F_UP_LR_HSI_01, F_PAN_01)   #(通道，空间)
        F_UP_LR_HSI_MD_01 = F_UP_LR_HSI_00 + F_MD_01["x_01"]
        F_PAN_MD_01 = F_PAN_00 + F_MD_01["x_02"]
        # 留出跳转连接变量
        F_UP_LR_HSI_MD_skip_01 = F_UP_LR_HSI_MD_01
        F_PAN_MD_skip_01 = F_PAN_MD_01
        #print("F_UP_LR_HSI_MD_skip_01.shape:--------------------->",F_UP_LR_HSI_MD_skip_01.shape)   # [8, 256, 80, 80]
        #print("F_PAN_MD_skip_01.shape:--------------------->",F_PAN_MD_skip_01.shape)   # [8, 256, 80, 80]
        #下采样
        F_UP_LR_HSI_01 = self.hs_down01(F_UP_LR_HSI_MD_01)  # [B, 4C, H/4, W/4]
        F_PAN_01 = self.pan_down01(F_PAN_MD_01)

        ''' ============================== 第二尺度/解码器 =============================== '''
        # 经过残差
        for i in range(self.unet_res_blocks[2]):
            F_UP_LR_HSI_01 = self.hs_res_block02[i](F_UP_LR_HSI_01)
        F_UP_LR_HSI_01 = self.hs_res_block02_conv(F_UP_LR_HSI_01)
        for i in range(self.unet_res_blocks[2]):
            F_PAN_01 = self.pan_res_block02[i](F_PAN_01)
        F_PAN_01 = self.pan_res_block02_conv(F_PAN_01)

        F_UP_LR_HSI_02 = F_UP_LR_HSI_01
        F_PAN_02 = F_PAN_01

        ''' # 曲率空间/通道注意力计算
        F_UP_LR_HSI_curvature_02 = self.img_curv(F_UP_LR_HSI_02)
        F_PAN_curvature_02 = self.img_curv(F_PAN_02)
        F_UP_LR_HSI_curvature_02 = self.curvature_CA_02(F_UP_LR_HSI_curvature_02)
        F_PAN_curvature_02 = self.curvature_SA_02(F_PAN_curvature_02)
        F_UP_LR_HSI_01 = F_UP_LR_HSI_curvature_02 * F_UP_LR_HSI_01    #加权计算
        F_PAN_01 = F_PAN_curvature_02 * F_PAN_01
        '''
        
        # 矩阵分解混合注意力计算
        #F_MD_02 = self.hamburger_02(F_PAN_02, F_UP_LR_HSI_02)   #(通道，空间)
        F_MD_02 = self.hamburger_02(F_UP_LR_HSI_02, F_PAN_02)   #(通道，空间)
        F_UP_LR_HSI_MD_02 = F_UP_LR_HSI_01 + F_MD_02["x_01"]
        F_PAN_MD_02 = F_PAN_01 + F_MD_02["x_02"]
        #上采样
        F_UP_LR_HSI_MD_02 = self.hs_up00(F_UP_LR_HSI_MD_02)  # [B, 2C, H/2, W/2]
        F_PAN_MD_02 = self.pan_up00(F_PAN_MD_02)        
        #print("F_UP_LR_HSI_MD_02.shape:--------------------->",F_UP_LR_HSI_MD_02.shape) # [8, 256, 80, 80]
        #print("F_PAN_MD_02.shape:--------------------->",F_PAN_MD_02.shape) # [8, 256, 80, 80]        
        
        # 加上跳转连接变量
        F_UP_LR_HSI_02 = F_UP_LR_HSI_MD_02 + F_UP_LR_HSI_MD_skip_01
        F_PAN_02 = F_PAN_MD_02 + F_PAN_MD_skip_01



        ''' =============================== 第一尺度/解码器 =================================== '''
        # 经过残差
        for i in range(self.unet_res_blocks[3]):
            F_UP_LR_HSI_02 = self.hs_res_block03[i](F_UP_LR_HSI_02)
        F_UP_LR_HSI_02 = self.hs_res_block03_conv(F_UP_LR_HSI_02)
        for i in range(self.unet_res_blocks[3]):
            F_PAN_02 = self.pan_res_block03[i](F_PAN_02)
        F_PAN_02 = self.pan_res_block03_conv(F_PAN_02)

        F_UP_LR_HSI_03 = F_UP_LR_HSI_02
        F_PAN_03 = F_PAN_02

        ''' # 曲率空间/通道注意力计算
        F_UP_LR_HSI_curvature_03 = self.img_curv(F_UP_LR_HSI_03)
        F_PAN_curvature_03 = self.img_curv(F_PAN_03)
        F_UP_LR_HSI_curvature_03 = self.curvature_CA_03(F_UP_LR_HSI_curvature_03)
        F_PAN_curvature_03 = self.curvature_SA_03(F_PAN_curvature_03)
        F_UP_LR_HSI_02 = F_UP_LR_HSI_curvature_03 * F_UP_LR_HSI_02  # 加权计算
        F_PAN_02 = F_PAN_curvature_03 * F_PAN_02
        '''
        
        # 矩阵分解混合注意力计算
        # F_MD_03 = self.hamburger_03(F_PAN_03, F_UP_LR_HSI_03)  # (通道，空间)
        F_MD_03 = self.hamburger_03(F_UP_LR_HSI_03, F_PAN_03)  # (通道，空间)
        F_UP_LR_HSI_MD_03 = F_UP_LR_HSI_02 + F_MD_03["x_01"]
        F_PAN_MD_03 = F_PAN_02 + F_MD_03["x_02"]

        # 上采样
        F_UP_LR_HSI_MD_03 = self.hs_up01(F_UP_LR_HSI_MD_03)  # [B, C, H, W]
        F_PAN_MD_03 = self.pan_up01(F_PAN_MD_03)        
        #print("F_UP_LR_HSI_MD_03.shape:--------------------->",F_UP_LR_HSI_MD_03.shape) # [8, 128, 160, 160] 
        #print("F_PAN_MD_03.shape:--------------------->",F_PAN_MD_03.shape) # [8, 128, 160, 160] 
        
        # 加上跳转连接变量
        F_UP_LR_HSI_03 = F_UP_LR_HSI_MD_03 + F_UP_LR_HSI_MD_skip_00
        F_PAN_03 = F_PAN_MD_03 + F_PAN_MD_skip_00


        ''' ======================== 最终的融合模块 ====================== '''
        F_fusion = torch.cat((F_PAN_03, F_UP_LR_HSI_03), dim = 1)  #[B, 2C, H, W]

        '''#初始残差 cat'''
        for i in range(self.unet_res_blocks[4]):
            F_fusion = self.fusion_res_block[i](F_fusion)
        F_fusion = self.fusion_res_block_conv(F_fusion)
        
        # 曲率空间/通道注意力计算
        F_fusion_ca = F_fusion
        F_fusion_sa = F_fusion  

        '''        
        F_fusion_curvature = self.img_curv(F_fusion_ca)
        F_fusion_curvature_ca = self.curvature_CA(F_fusion_curvature)
        F_fusion_curvature_sa = self.curvature_SA(F_fusion_curvature)
        #F_fusion_ca = F_fusion_curvature_ca * F_fusion    #加权计算
        #F_fusion_sa = F_fusion_curvature_sa * F_fusion
        F_fusion_ca = F_fusion_curvature_ca * F_UP_LR_HSI    #加权计算
        F_fusion_sa = F_fusion_curvature_sa * F_PAN     
        '''   

        # 矩阵分解混合注意力计算
        F_MD = self.Hamburger_final(F_fusion_ca, F_fusion_sa)   #(通道，空间)
        F_UP_LR_HSI_MD = F_fusion_ca + F_MD["x_01"]
        F_PAN_MD = F_fusion_sa + F_MD["x_02"]

        F_fusion_final = F_UP_LR_HSI_MD + F_PAN_MD

        F_out = self.compress(F_fusion_final)

        output = {"pred": F_out}

        return output
        