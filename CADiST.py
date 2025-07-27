import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat :int = 96,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    



class DsiFormer(nn.Module):
    
    def __init__(self, in_channels: int, embed_dim=64, num_heads=4, patch_size=4):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)
        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        )

        self.gate = nn.Parameter(torch.ones(1))


        self.slice_proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1)
        self.slice_unproj = nn.Conv1d(embed_dim, in_channels, kernel_size=1)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_norm2 = nn.LayerNorm(embed_dim)
        self.cross_mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        
        B, C, D, H, W = x.shape
        x_w = x.permute(0, 4, 1, 2, 3).reshape(B * W, C, D, H)

        # Patch embedding
        patch_feat = self.patch_embed(x_w)  # [B*W, embed_dim, D/ps, H/ps]
        B_, Edim, Hp, Wp = patch_feat.shape
        patch_feat = patch_feat.flatten(2).transpose(1, 2)  # [B*W, Hp*Wp, embed_dim]

        # Multi-head Self-Attention 
        skip = patch_feat
        patch_feat = self.norm1(patch_feat)
        attn_out, _ = self.mhsa(patch_feat, patch_feat, patch_feat)
        patch_feat = skip + attn_out

        # MLP 
        skip = patch_feat
        patch_feat = self.norm2(patch_feat)
        patch_feat = self.mlp(patch_feat)
        patch_feat = skip + patch_feat

        patch_feat = patch_feat.transpose(1, 2).view(B_, Edim, Hp, Wp)
        x_out = self.patch_unembed(patch_feat)
        x_out = x_out + self.gate * self.residual_connection(x_w)

        x_out = x_out.view(B, W, C, D, H).permute(0, 2, 3, 4, 1)  # [B, C, D, H, W]

        # --- Step 2 ---
        x_slice = x_out.mean(dim=(2, 3))  # [B, C, W]
        x_slice_proj = self.slice_proj(x_slice)  # [B, embed_dim, W]
        x_slice_proj = x_slice_proj.permute(0, 2, 1)  # [B, W, embed_dim]

        slice_token = self.cross_norm(x_slice_proj)
        cross_attn_out, _ = self.cross_mhsa(slice_token, slice_token, slice_token)
        slice_token = x_slice_proj + cross_attn_out
        slice_token = slice_token + self.cross_mlp(self.cross_norm2(slice_token))

        slice_token = slice_token.permute(0, 2, 1)  # [B, embed_dim, W]
        slice_token = self.slice_unproj(slice_token)  # [B, in_channels, W]
        slice_token = slice_token.unsqueeze(2).unsqueeze(2)
        x_out = x_out + slice_token

        return x_out
    

class CrossGate(nn.Module):
    
    def __init__(self, channels, hidden_dim=64):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        
        self.mlp_fc1 = nn.Linear(2 * channels, hidden_dim)
        self.mlp_fc2 = nn.Linear(hidden_dim, 2 * channels)
        
    def forward(self, a, p):
        B, C, D, H, W = a.shape

        a_pool = a.mean(dim=[2,3,4])  # [B, C]
        p_pool = p.mean(dim=[2,3,4])  # [B, C]

        x = torch.cat([a_pool, p_pool], dim=1)

        skip = x
        x = F.relu(self.mlp_fc1(x), inplace=True)   # [B, hidden_dim]
        x = self.mlp_fc2(x)                         # [B, 2C]
        x = x + skip                                
        
        gating_a, gating_p = torch.split(x, C, dim=1)
 
        gating_a = torch.sigmoid(gating_a)  # (B, C)
        gating_p = torch.tanh(gating_p)     # (B, C)

        a_new = a * gating_a.view(B, C, 1, 1, 1)
        p_new = p + gating_p.view(B, C, 1, 1, 1)
        
        return a_new, p_new

class CosM(nn.Module):
    '''Cohomological Spectro‑Mesh Block '''
    def __init__(self, in_channels: int, num_bands: int = 3, gating_hidden_dim: int = 64):
        super().__init__()
        self.num_bands = num_bands
        self.in_channels = in_channels

        self.cross_gates = nn.ModuleList([
            CrossGate(channels=in_channels, hidden_dim=gating_hidden_dim)
            for _ in range(num_bands)
        ])

        self.band_weights = nn.Parameter(torch.ones(num_bands))

        self.fuse_conv = nn.Conv3d(in_channels * num_bands, in_channels, kernel_size=3, padding=1)

        self.use_res = True

    def forward(self, x):
        B, C, D, H, W = x.shape

        ffted = torch.fft.fftn(x, dim=(2, 3, 4))
        ffted = torch.fft.fftshift(ffted, dim=(2, 3, 4))

        d_grid = torch.linspace(-0.5, 0.5, D, device=x.device).view(D, 1, 1).expand(D, H, W)
        h_grid = torch.linspace(-0.5, 0.5, H, device=x.device).view(1, H, 1).expand(D, H, W)
        w_grid = torch.linspace(-0.5, 0.5, W, device=x.device).view(1, 1, W).expand(D, H, W)
        radius = torch.sqrt(d_grid**2 + h_grid**2 + w_grid**2)

        band_edges = torch.linspace(0, 0.5, self.num_bands + 1, device=x.device)
        band_masks = [((radius >= band_edges[i]) & (radius < band_edges[i + 1])).unsqueeze(0).unsqueeze(0)
                      for i in range(self.num_bands)]

        real_part = ffted.real
        imag_part = ffted.imag
        mag = torch.sqrt(real_part**2 + imag_part**2 + 1e-9)
        phase = torch.atan2(imag_part, real_part + 1e-9)

        feat_cat = []
        for i in range(self.num_bands):
            mask_i = band_masks[i].expand(B, 1, D, H, W)

            mag_i = mag * mask_i
            phase_i = phase * mask_i

            mag_i, phase_i = self.cross_gates[i](mag_i, phase_i)

            real_band = mag_i * torch.cos(phase_i)
            imag_band = mag_i * torch.sin(phase_i)
            fft_band = torch.complex(real_band, imag_band)

            fft_band_shifted = torch.fft.ifftshift(fft_band, dim=(2, 3, 4))
            band_spatial = torch.fft.ifftn(fft_band_shifted, dim=(2, 3, 4)).real

            feat_cat.append(band_spatial)

        weighted_feats = [feat * self.band_weights[i] for i, feat in enumerate(feat_cat)]
        out_cat = torch.cat(weighted_feats, dim=1)

        out_fused = self.fuse_conv(out_cat)

        if self.use_res:
            out_fused = out_fused + x

        return out_fused

    
    
class CADiST(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        
        features: Sequence[int] = (24, 48, 96, 192, 384, 24),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)


        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout,feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout,feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout,feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout,feat=12)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        self.dsiformer_0 = DsiFormer(in_channels=fea[0], embed_dim=24, num_heads=3, patch_size=4)
        self.dsiformer_1 = DsiFormer(in_channels=fea[1], embed_dim=48, num_heads=3, patch_size=4)
        self.dsiformer_2 = DsiFormer(in_channels=fea[2], embed_dim=96, num_heads=3, patch_size=4)
        self.dsiformer_3 = DsiFormer(in_channels=fea[3], embed_dim=192, num_heads=3, patch_size=4)

        self.cosm_0 = CosM(in_channels=fea[0], num_bands=3, gating_hidden_dim=48)
        self.cosm_1 = CosM(in_channels=fea[1], num_bands=3, gating_hidden_dim=48)
        self.cosm_2 = CosM(in_channels=fea[2], num_bands=3, gating_hidden_dim=48)




    def forward(self, x: torch.Tensor):
                
        x0 = self.conv_0(x)

        x0 = self.dsiformer_0(x0)
               
        x1 = self.down_1(x0)

        x1 = self.dsiformer_1(x1)
                      
        x2 = self.down_2(x1)

        x2 = self.dsiformer_2(x2)
                
        x3 = self.down_3(x2)
        
        x3 = self.dsiformer_3(x3)

        x4 = self.down_4(x3)

        x0 = self.cosm_0(x0) 

        x1 = self.cosm_1(x1) 

        x2 = self.cosm_2(x2) 

        u4 = self.upcat_4(x4, x3)
        
        u3 = self.upcat_3(u4, x2)
        
        u2 = self.upcat_2(u3, x1)

        u1 = self.upcat_1(u2, x0)
        
        logits = self.final_conv(u1)

        return logits









