import torch
import torch.nn as nn
from net.transformer_utils import *
# 不需要导入 LCA，因为不再需要跨分支交互

class CIDNet(nn.Module):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        
        # 只保留 I 分支的编码器（Intensity Encoder）
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False) # 输入直接是3通道RGB
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        # 只保留 I 分支的解码器（Intensity Decoder）
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        # 跳过 HVI 转换，直接使用 RGB 输入
        i = x 
        
        # Encoder 阶段 (只保留 I 分支路径)
        i_enc1 = self.IE_block0(i)
        i_enc2 = self.IE_block1(i_enc1)
        i_enc3 = self.IE_block2(i_enc2)
        i_enc4 = self.IE_block3(i_enc3)
        
        # Decoder 阶段 (带 Skip Connection)
        i_dec3 = self.ID_block3(i_enc4, i_enc3) # 假设 NormUpsample 处理了拼接
        i_dec2 = self.ID_block2(i_dec3, i_enc2)
        i_dec1 = self.ID_block1(i_dec2, i_enc1)
        
        out = self.ID_block0(i_dec1)
        
        # 残差连接
        return out + x