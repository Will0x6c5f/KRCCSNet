import torch
import torch.nn.functional as F
from torch import nn
from .encoder_decoder import *
import pdb
torch.set_printoptions(precision=32)


def build_LKSN_ARM(sensing_rate):
    encoder=LKSN(sensing_rate)
    decoder=ARMDecoder(sensing_rate)
    csnet=EncoderDecoder(encoder,decoder)
    return csnet

def build_krccsnet(sensing_rate):
    if sensing_rate==0.5:
        encoder=nn.Conv2d(1, 2, kernel_size=(17, 17), stride=(2, 2), padding=(8, 8), bias=False)
    elif sensing_rate==0.25:
        encoder=nn.Conv2d(1, 4, kernel_size=(21, 21), stride=(4, 4), padding=(10, 10), bias=False)
    elif sensing_rate==0.125:
        encoder=nn.Conv2d(1, 2, kernel_size=(21, 21), stride=(4, 4), padding=(10, 10), bias=False)
    elif sensing_rate==0.0625:
        encoder=nn.Conv2d(1, 4, kernel_size=(29, 29), stride=(8, 8), padding=(14, 14), bias=False)
    elif sensing_rate==0.03125:
        encoder=nn.Conv2d(1, 2, kernel_size=(29, 29), stride=(8, 8), padding=(14, 14), bias=False)
    elif sensing_rate==0.015625:
        encoder=nn.Conv2d(1, 4, kernel_size=(45, 45), stride=(16, 16), padding=(22, 22), bias=False)

    decoder=ARMDecoder(sensing_rate)
    csnet=EncoderDecoder(encoder,decoder)
    return csnet


def conv1x1(in_ch: int, out_ch: int, stride: int = 1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class GateConv(nn.Module):
    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class Res3x3(nn.Module):
    def __init__(self, channel):
        super().__init__()        
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=2, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        res = self.pool(x)
        return  out+res


class LKSN(nn.Module):
    def __init__(self,sensing_rate=0.5):
        super().__init__()
     
        if sensing_rate==0.5:
            stride=1
            depth=1
            out_channel=2
        elif sensing_rate==0.25:
            stride=1
            depth=2
            out_channel=4
        elif sensing_rate==0.125:
            stride=1
            depth=2
            out_channel=2
        elif sensing_rate==0.0625:
            stride=1
            depth=3
            out_channel=4
        elif sensing_rate==0.03125:
            stride=1
            depth=3
            out_channel=2
        elif sensing_rate==0.015625:
            stride=1
            depth=4
            out_channel=4
        
        self.depth=depth
        c=32
       
        self.conv=nn.Conv2d(1, c, kernel_size=15, padding=7, stride=stride, bias=False)
        self.down=nn.Sequential( *[Res3x3(c) for i in range(depth)] )
        self.linear=nn.Conv2d(c, out_channels=out_channel, kernel_size=1, padding=0, stride=1, bias=False)
        

    def forward(self,img):
        x=self.conv(img)
        y=self.down(x)
        z=self.linear(y)
        
        return z
        
class ARMDecoder(nn.Module):
    def __init__(self, sensing_rate):
        super().__init__()

        self.sensing_rate = sensing_rate
    
        self.base = 64       
        self.res=nn.ModuleList()
        if sensing_rate == 0.5:
            self.initial = nn.Conv2d(2, 4, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 2
            for i in range(1,6):
                self.res.append(ARM(sensing_rate))
        elif sensing_rate == 0.25:
            self.initial = nn.Conv2d(4, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 4
            for i in range(1,6):
                self.res.append(ARM(sensing_rate))
        elif sensing_rate == 0.125:
            self.initial = nn.Conv2d(2, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 4
            for i in range(1,6):
                self.res.append(ARM(sensing_rate))
        elif sensing_rate == 0.0625:
            self.initial = nn.Conv2d(4, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 8
            for i in range(1,9):
                self.res.append(ARM(sensing_rate))
        elif sensing_rate == 0.03125:
            self.initial = nn.Conv2d(2, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 8
            for i in range(1,9):
                self.res.append(ARM(sensing_rate))
        elif sensing_rate == 0.015625:
            self.initial = nn.Conv2d(4, 256, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 16
            for i in range(1,9):
                self.res.append(ARM(sensing_rate))

        self.head = nn.Conv2d(1, self.base, kernel_size=3, padding=1, stride=1, bias=True)
        self.tail = nn.Conv2d(self.base, 1, kernel_size=3, padding=1, stride=1, bias=True)

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, sample):
        
        x = self.initial(sample)
        initial = nn.PixelShuffle(self.m)(x)

        out = self.ReLU(self.head(initial))
        for i,m in enumerate(self.res):
            out=m(out,sample)

        out = self.tail(out)

        return out + initial, initial

class ARM(nn.Module):
    def __init__(self,sensing_rate) -> None:
        super().__init__()
        # Notice K (in our paper) = self.total_depth - 2
        if sensing_rate==0.015625:
            self.total_depth=6
            self.stack=nn.ModuleList([
                UnetLayer(y_channel=4,gate_attn=False,depth=0,total_depth=6),
                UnetLayer(y_channel=4,gate_attn=False,depth=1,total_depth=6),
                UnetLayer(y_channel=4,gate_attn=False, depth=2,total_depth=6),
                UnetLayer(y_channel=4,gate_attn=True, depth=3,total_depth=6),
                UnetLayer(y_channel=4,gate_attn=True, depth=4,total_depth=6),
                UnetLayer(y_channel=4,gate_attn=True, depth=5,total_depth=6),
            ])
        elif sensing_rate==0.03125:
            self.total_depth=5
            self.stack=nn.ModuleList([
                UnetLayer(y_channel=2,gate_attn=False,depth=0,total_depth=5),
                UnetLayer(y_channel=2,gate_attn=False,depth=1,total_depth=5),
                UnetLayer(y_channel=2,gate_attn=True, depth=2,total_depth=5),
                UnetLayer(y_channel=2,gate_attn=True, depth=3,total_depth=5),
                UnetLayer(y_channel=2,gate_attn=True, depth=4,total_depth=5),
            ])
        elif sensing_rate==0.0625:
            self.total_depth=5
            self.stack=nn.ModuleList([
                UnetLayer(y_channel=4,gate_attn=False,depth=0,total_depth=5),
                UnetLayer(y_channel=4,gate_attn=False,depth=1,total_depth=5),
                UnetLayer(y_channel=4,gate_attn=True, depth=2,total_depth=5),
                UnetLayer(y_channel=4,gate_attn=True, depth=3,total_depth=5),
                UnetLayer(y_channel=4,gate_attn=True, depth=4,total_depth=5),
            ])
        elif sensing_rate==0.125:
            self.total_depth=4
            self.stack=nn.ModuleList([
                UnetLayer(y_channel=2,gate_attn=False,depth=0,total_depth=4),
                UnetLayer(y_channel=2,gate_attn=False,depth=1,total_depth=4),
                UnetLayer(y_channel=2,gate_attn=True, depth=2,total_depth=4),
                UnetLayer(y_channel=2,gate_attn=True, depth=3,total_depth=4),
            ])
        elif sensing_rate==0.25:
            self.total_depth=4
            self.stack=nn.ModuleList([
                UnetLayer(y_channel=4,gate_attn=False,depth=0,total_depth=4),
                UnetLayer(y_channel=4,gate_attn=False,depth=1,total_depth=4),
                UnetLayer(y_channel=4,gate_attn=True, depth=2,total_depth=4),
                UnetLayer(y_channel=4,gate_attn=True, depth=3,total_depth=4),
            ])
        elif sensing_rate==0.5:
            self.total_depth=3
            self.stack=nn.ModuleList([
                UnetLayer(y_channel=2,gate_attn=False,depth=0,total_depth=3),
                UnetLayer(y_channel=2,gate_attn=False,depth=1,total_depth=3),
                UnetLayer(y_channel=2,gate_attn=True, depth=2,total_depth=3),
            ])
        
    def forward(self,x,y):
        
        h_stack=[]
        h=x
        for i in range(self.total_depth):
            h=self.stack[i].down(h)
            h_stack.append(h)
        o=None
        for i in range(self.total_depth-1,0,-1):
            o=self.stack[i].up(h_stack.pop(),o,y)
        o=self.stack[0].up(h_stack.pop(),o,None)

        return o


class UnetLayer(nn.Module):
    def __init__(self,y_channel=2,gate_attn=False,depth=1,total_depth=3) -> None:
        super().__init__()
        self.depth=depth

        self.catconv=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )

        if depth==total_depth-1:
            self.downconv=None
            self.mix=nn.Sequential(
                nn.Conv2d(y_channel, 1, kernel_size=1, padding=0, stride=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
                nn.ReLU(),
            )
            self.catconv_ps=None
        elif depth==0:
            self.downconv=None
            self.mix=None
            self.catconv_ps=None
        else:
            self.downconv=nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
                nn.ReLU()
            )
            self.mix = nn.Sequential(
                nn.Conv2d(y_channel, 2**(2*(total_depth-depth-1)), kernel_size=1, padding=0, stride=1, bias=True),
                nn.ReLU(),
                nn.PixelShuffle(2**(total_depth-depth-1)),
                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
                nn.ReLU(),
            )
            self.catconv_ps=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
                nn.ReLU(),
                nn.PixelShuffle(2)
            )

        if gate_attn:
            self.gate_attn=GateConv(64)
        else:
            self.gate_attn=None
        self.depth=depth

    def down(self,h):
        if self.downconv is not None:
            h=self.downconv(h)
        else :
            h=h
        return h

    def up(self,h,o=None,y=None):
        if o is not None:
            if self.catconv_ps is not None:
                o=self.catconv_ps(torch.cat([o, h], dim=1))
            else :
                o=self.catconv(torch.cat([o, h], dim=1))
        else:
            o=h

        if y is not None:
            y_mix=self.mix(y)
            o=self.catconv(torch.cat([o, y_mix], dim=1))
            if self.gate_attn is not None:
                o=self.gate_attn(o)
            
        return o


