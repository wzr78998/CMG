import random

import torch
import math
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.autograd as ag
class GELU(nn.Module):#zengen
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        #return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class GroupAttention(nn.Module):
    def __init__(self, dim, num_heads=4, N_Pi=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1,
                 sr_ratio=1.0):
        """
        ws 1 for stand attention
        """
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        dim = dim // N_Pi  # 光谱token数
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * N_Pi, dim * N_Pi)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    # @auto_fp16()
    def forward(self, x, N_Pi, D_Pi):
        B, N, C = x.shape
        x = x.view(B, N, N_Pi, D_Pi)
        qkv = self.qkv(x).reshape(B, N, N_Pi, 3, self.num_heads,
                                  D_Pi // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3)
        x = attn.reshape(B, N, N_Pi, D_Pi)
        # if pad_r > 0 or pad_b > 0:
        #     x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        # dim = dim // 4
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

        # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # x1 = x.reshape(B,-1,16)
        # B, N, C = x1.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        # x_ = self.sr(x)
        x_ = self.norm(x)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.drop_path = nn.Identity()  # zhan wei
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class GroupBlock(Block):
    def __init__(self, depth, dim, num_heads, N_Pi, patch_size, local_kiner=3, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        # del self.attn1
        # del self.attn2
        # if ws == 1:
        #     self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        # else:
        #     self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)
        self.depth = depth
        self.attn1 = GroupAttention(dim, num_heads, N_Pi, qkv_bias, qk_scale, attn_drop, drop)
        self.attn2 = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        self.pool = nn.AvgPool2d(local_kiner, 1, (local_kiner - 1) // 2)

        self.patch_size = patch_size
        self.dim = dim



    def forward(self, x, N_Pi, D_Pi, H, W,  GPU=0):
        for i in range(self.depth):
            if i < 0:
                x = x + self.drop_path(self.attn2(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            else:
                x = x + self.drop_path(self.attn1(self.norm1(x), N_Pi, D_Pi))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                x = x + self.drop_path(self.attn2(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                x = x + self.drop_path(self.attn2(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
class FE(nn.Module):
    def __init__(self, image_size, near_band, num_patches, patch_size, num_classes, dim, pixel_dim, depth, heads,
                 mlp_dim, pool='cls', channels=1, dim_head=16, dropout=0., emb_dropout=0., mode='ViT', GPU=1,
                 local_kiner=3):
        super().__init__()

        patch_dim = image_size

        self.GPU = GPU
        self.patch_size = patch_size
        self.dim = dim

        self.num_classes = num_classes

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding_p = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dis_cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = GroupBlock(depth=depth, dim=dim, num_heads=heads, N_Pi=pixel_dim, patch_size=self.patch_size,
                                      local_kiner=local_kiner)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):

        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]

        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]

        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding_p[:, :]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # x = rearrange(x, 'b n d -> (b n) d')
        # x = x.reshape(-1,16,4)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        N_Pi = 4
        D_Pi = 16
        H = W = 14
        casual= self.transformer(x, N_Pi, D_Pi, H, W,  self.GPU)
        # classification: using cls_token output
        x = self.to_latent(casual[:, 0])

        # attention module

        return x





class C_Encoder (nn.Module):
    def __init__(self, dim,e_num,euc_num,d_num):
        super().__init__()
        self.e_num=e_num
        self.euc_num=euc_num
        self.d_num=d_num





        self.dim = dim
        self.layer1=nn.Sequential()
        for i in range (self.e_num):
            self.layer1.append(nn.LayerNorm(dim))
            self.layer1.append(nn.Linear(dim,dim))
            self.layer1.append(nn.Tanh())





        self.layer2 = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim, dim))
        self.layer3 = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim, dim))
        self.Tanh=nn.Tanh()





    def forward(self, x):
        x=self.layer1(x)


        mean = self.layer2(x)

        var = self.layer3(x)

        x=torch.cat([mean,var],1)





        return x


            # add position embedding

class UC_Encoder (nn.Module):
    def __init__(self, dim,e_num,euc_num,d_num):
        super().__init__()
        self.e_num = e_num
        self.euc_num = euc_num
        self.d_num = d_num





        self.dim = dim

        self.layer1 = nn.Sequential()
        for i in range(self.euc_num):
            self.layer1.append(nn.LayerNorm(dim))
            self.layer1.append(nn.Linear(dim, dim))
            self.layer1.append(nn.Tanh())
        self.layer2 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.layer3 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.Tanh=nn.Tanh()





    def forward(self, x):
        x = self.layer1(x)


        mean = self.layer2(x)

        var = self.layer3(x)


        x = torch.cat([mean, var], 1)

        return x
class Decoder (nn.Module):
    def __init__(self, dim, e_num, euc_num, d_num):
        super().__init__()
        self.e_num = e_num
        self.euc_num = euc_num
        self.d_num = d_num
        self.layer1 = nn.Sequential(nn.LayerNorm(dim*2),nn.Linear(dim*2,dim*2),nn.Tanh())
        for i in range(self.d_num):
            self.layer1.append(nn.LayerNorm(dim*2))
            self.layer1.append(nn.Linear(dim*2, dim*2))
            self.layer1.append(nn.Tanh())
        self.layer1.append((nn.LayerNorm(dim*2)))
        self.layer1.append(nn.Linear(dim*2, dim))





        self.dim = dim












    def forward(self, x):

        x = self.layer1(x)
        return x



class B_VAE_1(nn.Module):
    def __init__(self, dim,e_num,euc_num,d_num):
        super().__init__()





        self.dim = dim




        self.c_encoder =C_Encoder(dim,e_num,euc_num,d_num)
        self.uc_encoder = UC_Encoder(dim,e_num,euc_num,d_num)
        self.decoder =Decoder(dim,e_num,euc_num,d_num)




    def forward(self, x, label=0):

        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]

        if label==0:


            # add position embedding

            casual = self.c_encoder(x)
            # classification: using cls_token output
            mean_c =casual[:,:self.dim]
            var_c = casual[:,self.dim:]
            un_casual = self.uc_encoder(x)
            # classification: using cls_token output
            mean_uc = un_casual[:, :self.dim]
            var_uc = un_casual[:, self.dim:]
            z_c = mean_c + torch.randn_like(mean_c) * var_c
            z_uc = mean_uc + torch.randn_like(mean_uc) * var_uc
            rec_x = self.decoder(torch.cat([z_c,z_uc],1))
            return mean_c, mean_uc, var_c, var_uc, z_c, z_uc, rec_x
        if label==1:
            casual = self.c_encoder(x)
            # classification: using cls_token output
            mean_c = casual[:, :self.dim]
            var_c = casual[:, self.dim:]
            z_c = mean_c + torch.randn_like(mean_c) * var_c
            return mean_c, var_c,  z_c
        if label == 2:

            rec_x = self.decoder(x)
            return rec_x