import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *

class CIDNet(nn.Module):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # --- HV 分支 (Color Branch) ---
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

        # --- I 分支 (Intensity Branch) ---
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False)
        )

        # --- 注意力层 (自注意力版) ---
        # 我们依然使用原代码中的 CAB，但在 forward 中只给它传同一个分支的特征
        self.I_SelfAttn1 = CAB(ch2, head2, bias=False)
        self.HV_SelfAttn1 = CAB(ch2, head2, bias=False)
        self.I_SelfAttn2 = CAB(ch3, head3, bias=False)
        self.HV_SelfAttn2 = CAB(ch3, head3, bias=False)
        self.I_SelfAttn3 = CAB(ch4, head4, bias=False)
        self.HV_SelfAttn3 = CAB(ch4, head4, bias=False)
        
        self.trans = RGB_HVI()

    def forward(self, x):
        # HVI 空间转换
        dtypes = x.dtype
        hv = self.trans.HVIT(x)
        i = hv[:,2,:,:].unsqueeze(1).to(dtypes)
        
        # Encoder 阶段
        i_1 = self.IE_block0(i)
        hv_1 = self.HVE_block0(hv)
        
        i_2 = self.IE_block1(i_1)
        hv_2 = self.HVE_block1(hv_1)
        
        # 消融核心：只做自注意力，不进行跨分支交换
        # 原版是 self.I_LCA(i, hv)，这里改为 self.I_SelfAttn(i, i)
        i_enc2 = self.I_SelfAttn1(i_2, i_2) 
        hv_enc2 = self.HV_SelfAttn1(hv_2, hv_2)
        
        i_3 = self.IE_block2(i_enc2)
        hv_3 = self.HVE_block2(hv_enc2)
        
        i_enc3 = self.I_SelfAttn2(i_3, i_3)
        hv_enc3 = self.HV_SelfAttn2(hv_3, hv_3)
        
        i_4 = self.IE_block3(i_enc3)
        hv_4 = self.HVE_block3(hv_enc3)
        
        i_enc4 = self.I_SelfAttn3(i_4, i_4)
        hv_enc4 = self.HV_SelfAttn3(hv_4, hv_4)
        
        # Decoder 阶段 (保持双分支独立)
        i_dec3 = self.ID_block3(i_enc4, i_enc3)
        hv_dec3 = self.HVD_block3(hv_enc4, hv_enc3)
        
        i_dec2 = self.ID_block2(i_dec3, i_enc2)
        hv_dec2 = self.HVD_block2(hv_dec3, hv_enc2)
        
        i_dec1 = self.ID_block1(i_dec2, i_1)
        hv_dec1 = self.HVD_block1(hv_dec2, hv_1)
        
        i_out = self.ID_block0(i_dec1)
        hv_out = self.HVD_block0(hv_dec1)
        
        output_hvi = torch.cat([hv_out, i_out], dim=1) + hv 
        output_rgb = self.trans.PHVIT(output_hvi)
        
        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi