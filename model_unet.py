from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x):
        b,c,h,w = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(b,c,-1).transpose(1,2)
        k = k.view(b,c,-1).transpose(1,2)
        v = v.view(b,c,-1).transpose(1,2)

        attn = torch.bmm(q, k.transpose(1,2))/(self.in_channels**0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)

        out = out.transpose(1,2).view(b,c,h,w)

        out = self.gamma * out + x
        out = self.norm(out)

        return out

class ConditionalUNet(nn.Module):
    def __init__(self, time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.inc = DoubleConv(2, 64)
        
        self.down1 = Down(64, 128)
        self.attn1 = SelfAttention2D(128)

        self.down2 = Down(128, 256)
        self.attn2 = SelfAttention2D(256)

        self.down3 = Down(256, 256)
        self.attn3 = SelfAttention2D(256)

        self.bot1 = DoubleConv(256+time_emb_dim, 256)
        self.bot_attn = SelfAttention2D(256)
        self.bot2 = DoubleConv(256, 256)

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)

        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t, cond):
        x = torch.cat([x, cond], dim=1)

        t_emb = self.time_mlp(t.unsqueeze(-1))

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x2 = self.attn1(x2)
        
        x3 = self.down2(x2)
        x3 = self.attn2(x3)

        x4 = self.down3(x3)
        x4 = self.attn3(x4)

        t_expand = t_emb[:, :, None, None].repeat(1, 1, x4.shape[2], x4.shape[3])
        x4 = torch.cat([x4, t_expand], dim=1)
        
        x4 = self.bot1(x4)
        x4 = self.bot_attn(x4)
        x4 = self.bot2(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return self.outc(x)
