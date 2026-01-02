import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """
    Attention Gate to focus on relevant features from the skip connection.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in the gating signal (from decoder)
            F_l: Number of channels in the skip connection (from encoder)
            F_int: Number of intermediate channels
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 if necessary (due to pooling/striding differences)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class DecoderBlock(nn.Module):
    """
    Decoder Block with Attention Gate.
    Upsamples input, applies attention to skip connection, concatenates, and applies ConvBlock.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Attention Gate
        self.att = AttentionBlock(F_g=in_channels, F_l=skip_channels, F_int=in_channels // 2)
        
        # ConvBlock after concatenation
        # Input channels = in_channels (from upsampled) + skip_channels (from attention)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch if any (padding/cropping issues)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
            
        skip = self.att(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class SiameseUnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(SiameseUnet, self).__init__()
        
        # 1. Encoder: Pre-trained ResNet-34
        # We will use the layers to extract features at different scales
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        self.firstconv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
            # Maxpool is usually applied here in ResNet, we'll handle it carefully
        )
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # Feature channels for ResNet34
        # layer1: 64, layer2: 128, layer3: 256, layer4: 512
        # firstconv output: 64
        
        # 2. Decoder
        # We fuse features from pre and post, so channels double
        
        # Bottleneck (Center)
        # Input comes from encoder4 (512) * 2 (pre+post) = 1024
        self.center = ConvBlock(1024, 512)
        
        # Decoder Blocks
        # d4: takes center (512) and skip3 (256*2=512) -> out 256
        self.d4 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        
        # d3: takes d4 (256) and skip2 (128*2=256) -> out 128
        self.d3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        
        # d2: takes d3 (128) and skip1 (64*2=128) -> out 64
        self.d2 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)
        
        # d1: takes d2 (64) and skip0 (64*2=128) -> out 64
        # skip0 is from firstconv (before maxpool)
        self.d1 = DecoderBlock(in_channels=64, skip_channels=128, out_channels=64)
        
        # Final upsampling to restore original resolution (stride 2 -> stride 1)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 3. Heads
        # mask_loc: Building vs Background
        self.head_loc = nn.Conv2d(32, n_classes, kernel_size=1)
        
        # mask_clf: Damaged vs Undamaged
        self.head_clf = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward_encoder(self, x):
        # x: (B, 3, H, W)
        x0 = self.firstconv(x)      # (B, 64, H/2, W/2)
        x_mp = self.maxpool(x0)     # (B, 64, H/4, W/4)
        
        x1 = self.encoder1(x_mp)    # (B, 64, H/4, W/4)
        x2 = self.encoder2(x1)      # (B, 128, H/8, W/8)
        x3 = self.encoder3(x2)      # (B, 256, H/16, W/16)
        x4 = self.encoder4(x3)      # (B, 512, H/32, W/32)
        
        return x0, x1, x2, x3, x4

    def forward(self, x_pre, x_post):
        # Encoder - Shared Weights
        pre_0, pre_1, pre_2, pre_3, pre_4 = self.forward_encoder(x_pre)
        post_0, post_1, post_2, post_3, post_4 = self.forward_encoder(x_post)
        
        # Feature Fusion (Concatenation)
        f4 = torch.cat([pre_4, post_4], dim=1) # 512+512 = 1024
        f3 = torch.cat([pre_3, post_3], dim=1) # 256+256 = 512
        f2 = torch.cat([pre_2, post_2], dim=1) # 128+128 = 256
        f1 = torch.cat([pre_1, post_1], dim=1) # 64+64 = 128
        f0 = torch.cat([pre_0, post_0], dim=1) # 64+64 = 128
        
        # Decoder
        center = self.center(f4) # 1024 -> 512
        
        d4 = self.d4(center, f3) # 512, 512 -> 256
        d3 = self.d3(d4, f2)     # 256, 256 -> 128
        d2 = self.d2(d3, f1)     # 128, 128 -> 64
        d1 = self.d1(d2, f0)     # 64, 128 -> 64
        
        # Final Upsample
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        # Heads
        mask_loc = self.head_loc(out)
        mask_clf = self.head_clf(out)
        
        return mask_loc, mask_clf

if __name__ == "__main__":
    # Test the model
    model = SiameseUnet()
    x_pre = torch.randn(1, 3, 256, 256)
    x_post = torch.randn(1, 3, 256, 256)
    
    loc, clf = model(x_pre, x_post)
    
    print(f"Input shape: {x_pre.shape}")
    print(f"Output Loc shape: {loc.shape}")
    print(f"Output Clf shape: {clf.shape}")
