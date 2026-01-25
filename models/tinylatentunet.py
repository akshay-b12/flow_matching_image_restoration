import torch
import torch.nn as nn

from arch_utils import SinusoidalTimeEmb, ResBlock

class TinyLatentUNet(nn.Module):
    def __init__(self, in_ch=8, out_ch=4, base=64, tdim=256):
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalTimeEmb(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        # Down
        self.b1 = ResBlock(in_ch, base, tdim)         # 64x64
        self.down1 = nn.Conv2d(base, base, 4, 2, 1)   # 32x32
        self.b2 = ResBlock(base, base*2, tdim)        # 32x32
        self.down2 = nn.Conv2d(base*2, base*2, 4, 2, 1)  # 16x16

        self.mid = ResBlock(base*2, base*2, tdim)     # 16x16

        # Up
        self.up2 = nn.ConvTranspose2d(base*2, base*2, 4, 2, 1)  # 32x32
        self.ub2 = ResBlock(base*2 + base*2, base, tdim)        # skip with b2
        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)      # 64x64
        self.ub1 = ResBlock(base + base, base, tdim)            # skip with b1

        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, zt, t, cond):
        x = torch.cat([zt, cond], dim=1)   # [B,8,64,64]
        temb = self.time(t)               # [B,tdim]

        h1 = self.b1(x, temb)             # [B,base,64,64]
        d1 = self.down1(h1)               # [B,base,32,32]
        h2 = self.b2(d1, temb)            # [B,2b,32,32]
        d2 = self.down2(h2)               # [B,2b,16,16]

        m  = self.mid(d2, temb)           # [B,2b,16,16]

        u2 = self.up2(m)                  # [B,2b,32,32]
        u2 = self.ub2(torch.cat([u2, h2], dim=1), temb)  # [B,b,32,32]
        u1 = self.up1(u2)                 # [B,b,64,64]
        u1 = self.ub1(torch.cat([u1, h1], dim=1), temb)  # [B,b,64,64]

        return self.out(u1)               # [B,4,64,64]
