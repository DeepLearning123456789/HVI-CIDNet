import torch
import torch.nn as nn
import torch.nn.functional as F
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import HV_LCA, I_LCA, IEL, PAB, RRB 
from huggingface_hub import PyTorchModelHubMixin
import torch.fft

class FourierLowPassAttention(nn.Module):
    def __init__(self, in_channels):
        super(FourierLowPassAttention, self).__init__()
        self.frequency_dw = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='ortho')
        x_real = x_freq.real
        x_imag = x_freq.imag
        x_complex_concat = torch.cat([x_real, x_imag], dim=1)
        
        freq_weights = self.frequency_dw(x_complex_concat)
        x_complex_concat = x_complex_concat * freq_weights
        
        x_real_out, x_imag_out = torch.chunk(x_complex_concat, 2, dim=1)
        x_freq_out = torch.complex(x_real_out, x_imag_out)
        x_spatial = torch.fft.irfft2(x_freq_out, s=(H, W), norm='ortho')
        return x_spatial

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        self.fourier_attn = FourierLowPassAttention(in_channels=ch1)
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )
        
        # 升级 PAB 替换 LCA
        self.HV_PAB1 = PAB(ch2, head2)
        self.HV_PAB2 = PAB(ch3, head3)
        self.HV_PAB3 = PAB(ch4, head4)
        self.HV_PAB4 = PAB(ch4, head4)
        self.HV_PAB5 = PAB(ch3, head3)
        self.HV_PAB6 = PAB(ch2, head2)
        
        self.I_PAB1 = PAB(ch2, head2)
        self.I_PAB2 = PAB(ch3, head3)
        self.I_PAB3 = PAB(ch4, head4)
        self.I_PAB4 = PAB(ch4, head4)
        self.I_PAB5 = PAB(ch3, head3)
        self.I_PAB6 = PAB(ch2, head2)
        
        self.HV_iel = IEL(ch2)
        self.I_iel = IEL(ch2)
        
        # 最末端解码输出注入 RRB 块
        self.I_rrb = RRB(ch1)
        self.HV_rrb = RRB(ch1)
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        
        # 转换至 HVI 空间
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        # I 分支基础处理 & 傅里叶去噪
        i_enc0 = self.IE_block0(i)
        i_enc0_denoised = self.fourier_attn(i_enc0)
        i_enc0 = i_enc0 + i_enc0_denoised 
        
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = i_enc1 + self.I_PAB1(i_enc1, hv_1, i_enc1)
        hv_2 = hv_1 + self.HV_PAB1(hv_1, i_enc1, i_enc1)
        i_enc2 = self.I_iel(i_enc2)
        hv_2 = self.HV_iel(hv_2)
        
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = i_enc2 + self.I_PAB2(i_enc2, hv_2, i_enc2)
        hv_3 = hv_2 + self.HV_PAB2(hv_2, i_enc2, i_enc2)
        
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc3)
        hv_3 = self.HVE_block3(hv_3)
        
        # 瓶颈层交互 (Bottleneck)
        i_enc4 = i_enc3 + self.I_PAB3(i_enc3, hv_3, i_enc3)
        hv_4 = hv_3 + self.HV_PAB3(hv_3, i_enc3, i_enc3)
        
        # ---- 开始解码器阶段 ----
        i_dec4 = i_enc4 + self.I_PAB4(i_enc4, hv_4, i_enc4)
        hv_4 = hv_4 + self.HV_PAB4(hv_4, i_enc4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        
        i_dec2 = i_dec3 + self.I_PAB5(i_dec3, hv_3, i_dec3)
        hv_2 = hv_3 + self.HV_PAB5(hv_3, i_dec3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec2, v_jump1)
        
        i_dec1 = i_dec2 + self.I_PAB6(i_dec2, hv_2, i_dec2)
        hv_1 = hv_2 + self.HV_PAB6(hv_2, i_dec2, i_dec2)
        
        # 调用 RRB 进行照度不均匀自适应提亮和细节精细化。
        i_dec1 = self.I_rrb(i_dec1)
        hv_1 = self.HV_rrb(hv_1)
        
        # 精细化调节完成后，再安全地降维压缩到 1 通道和 2 通道
        i_dec0 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec0)
        
        hv_0 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_0)
        
        # 重构回 RGB 空间
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi