import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *

class CIDNet(nn.Module):
    def __init__(self, channels=[36, 36, 72, 144], heads=[1, 2, 4, 8], norm=False):
        super(CIDNet, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        
        # 输入层改为 3 通道
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        # Self-Attention 层
        self.I_SelfAttn1 = CAB(ch2, heads[1], bias=False)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        
        # 输出层改为 3 通道
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # 计算需要填充的量，确保长宽能被 16 整除（因为模型有 4 层下采样）
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # --- 2. 正常前向传播 (sRGB 路径) ---
        # Encoder
        i_1 = self.IE_block0(x)
        i_2 = self.IE_block1(i_1)
        
        # 执行自注意力消融
        i_2 = self.I_SelfAttn1(i_2, i_2) 
        
        i_3 = self.IE_block2(i_2)
        i_4 = self.IE_block3(i_3)
        
        # Decoder (此时尺寸都是 16 的倍数，绝对不会报错)
        i_dec3 = self.ID_block3(i_4, i_3)
        i_dec2 = self.ID_block2(i_dec3, i_2)
        i_dec1 = self.ID_block1(i_dec2, i_1)
        
        out = self.ID_block0(i_dec1)
        
        # --- 3. 还原尺寸并添加残差 ---
        res = out + x
        # 如果之前填充了，现在裁掉
        if pad_h > 0 or pad_w > 0:
            res = res[:, :, :H, :W]
            
        return res