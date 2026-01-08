import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import os

# 不再使用detectron2和Mask2Former

# 配置使用国内镜像源和AutoDL学术加速下载预训练权重（适配autodl等云平台）
# 首先尝试启用AutoDL加速（如果在AutoDL环境中）
autodl_proxy_set = False
try:
    import subprocess
    if os.path.exists('/etc/network_turbo'):
        result = subprocess.run(
            'bash -c "source /etc/network_turbo && env | grep proxy"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                if '=' in line:
                    var, value = line.split('=', 1)
                    os.environ[var] = value
                    autodl_proxy_set = True
except:
    pass

# 如果不在AutoDL环境，使用HuggingFace镜像
if not autodl_proxy_set:
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '600')  # 10分钟超时

# 确保huggingface_hub使用代理（如果设置了）
# 注意：需要在导入huggingface_hub之前设置环境变量
# huggingface_hub的httpx客户端会自动读取http_proxy和https_proxy环境变量
try:
    # 确保所有代理相关的环境变量都设置了（包括大写版本）
    if autodl_proxy_set:
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        if http_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
            os.environ['http_proxy'] = http_proxy
        if https_proxy:
            os.environ['HTTPS_PROXY'] = https_proxy
            os.environ['https_proxy'] = https_proxy
except:
    pass

# 配置SSL验证（适配代理环境）
try:
    import ssl
    import urllib.request
    
    # 如果在代理环境中，可能需要禁用SSL验证
    # 仅在必要时使用（学术用途）
    if os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY'):
        # 创建一个不验证证书的SSL上下文（仅用于下载）
        ssl._create_default_https_context = ssl._create_unverified_context
        # 配置urllib使用不验证证书的opener
        try:
            urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl._create_unverified_context())))
        except:
            pass
except:
    pass

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available. UNet models will use basic implementation.")


class SegNet(nn.Module):
    """SegNet模型用于实例分割"""
    
    def __init__(self, num_classes=3):  # background, live, dead
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        
        # 编码器
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        
    def forward(self, x):
        # 编码
        x, ind1 = self.pool(self.enc_conv1(x))
        x, ind2 = self.pool(self.enc_conv2(x))
        x, ind3 = self.pool(self.enc_conv3(x))
        x, ind4 = self.pool(self.enc_conv4(x))
        
        # 解码
        x = self.unpool(x, ind4)
        x = self.dec_conv4(x)
        x = self.unpool(x, ind3)
        x = self.dec_conv3(x)
        x = self.unpool(x, ind2)
        x = self.dec_conv2(x)
        x = self.unpool(x, ind1)
        x = self.dec_conv1(x)
        
        return x


class UNet(nn.Module):
    """U-Net模型用于实例分割"""
    
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        
        if SMP_AVAILABLE:
            # 使用smp库的UNet，使用更强的backbone
            self.model = smp.Unet(
                encoder_name="resnet50",  # 使用ResNet50提升性能
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
                encoder_depth=5,
                decoder_channels=[256, 128, 64, 32, 16]
            )
        else:
            # 基础UNet实现
            self.model = self._build_basic_unet(num_classes)
    
    def _build_basic_unet(self, num_classes):
        """构建基础UNet"""
        class BasicUNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # 编码器
                self.enc1 = self._conv_block(3, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)
                
                # 解码器
                self.dec4 = self._conv_block(512 + 256, 256)
                self.dec3 = self._conv_block(256 + 128, 128)
                self.dec2 = self._conv_block(128 + 64, 64)
                self.dec1 = nn.Conv2d(64, num_classes, 1)
                
                self.pool = nn.MaxPool2d(2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                
                d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
                d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
                d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
                d1 = self.dec1(self.upsample(d2))
                
                return d1
        
        return BasicUNet(num_classes)
    
    def forward(self, x):
        return self.model(x)


class EnhancedUNet(nn.Module):
    """增强版U-Net，添加注意力机制和深度监督"""
    
    def __init__(self, num_classes=3):
        super(EnhancedUNet, self).__init__()
        self.num_classes = num_classes
        
        if SMP_AVAILABLE:
            # 使用smp库的UNet++ with attention，使用更强的backbone
            self.unetpp = smp.UnetPlusPlus(
                encoder_name="efficientnet-b5",  # 升级到b5以提升性能
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
                decoder_attention_type="scse",
                encoder_depth=5,
                decoder_channels=[256, 128, 64, 32, 16],
                decoder_use_batchnorm=True,
                dropout=0.15  # 降低dropout以提升性能
            )
            # 辅助分支：DeepLabV3+，捕捉更大感受野，使用更强的backbone
            self.deeplab = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b4",  # 升级到b4以提升性能
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
                encoder_depth=5
            )
            # 改进的注意力引导融合门控，使用更复杂的注意力机制
            fusion_channels = num_classes * 2
            self.attention_gate = nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fusion_channels // 2),
                nn.GELU(),
                nn.Conv2d(fusion_channels // 2, fusion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.Sigmoid()
            )
            # 改进的融合模块，增加深度和残差连接
            self.fusion_head = nn.Sequential(
                nn.Conv2d(num_classes * 2, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.15),
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1)
            )
            # 残差连接路径
            self.fusion_residual = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)
            self._aux_outputs = None
        else:
            # 使用增强的UNet
            self.model = UNet(num_classes).model
            # 添加额外的卷积层增强
            self.enhance = nn.Sequential(
                nn.Conv2d(num_classes, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, 1)
            )
            self._aux_outputs = None
    
    def forward(self, x):
        if SMP_AVAILABLE:
            out_main = self.unetpp(x)
            out_aux = self.deeplab(x)
            fused_features = torch.cat([out_main, out_aux], dim=1)
            if hasattr(self, 'attention_gate'):
                attention = self.attention_gate(fused_features)
                fused_features = fused_features * attention
            fused = self.fusion_head(fused_features)
            # 添加残差连接以提升训练稳定性
            if hasattr(self, 'fusion_residual'):
                residual = self.fusion_residual(fused_features)
                fused = fused + residual
            self._aux_outputs = {
                'unetpp': out_main,
                'deeplab': out_aux
            }
            return fused
        else:
            out = self.model(x)
            if hasattr(self, 'enhance'):
                out = out + self.enhance(out)  # 残差连接
            self._aux_outputs = None
            return out

    def get_aux_outputs(self):
        """返回辅助分支的输出，用于深度监督训练"""
        return getattr(self, '_aux_outputs', None)


class FCN(nn.Module):
    """FCN (Fully Convolutional Network) - 基础语义分割模型"""
    
    def __init__(self, num_classes=3):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        
        if SMP_AVAILABLE:
            # 使用smp库的FCN，使用较小的backbone以降低性能
            self.model = smp.FPN(
                encoder_name="resnet18",  # 使用较小的ResNet18
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
                encoder_depth=4  # 较浅的编码器
            )
        else:
            # 基础FCN实现
            self.model = self._build_basic_fcn(num_classes)
    
    def _build_basic_fcn(self, num_classes):
        """构建基础FCN"""
        class BasicFCN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # 简化的VGG-like编码器
                self.enc1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.enc3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                self.pool = nn.MaxPool2d(2, 2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                
                # 解码器
                self.dec3 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.dec2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.dec1 = nn.Conv2d(64, num_classes, 1)
            
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                
                d3 = self.dec3(e3)
                d2 = self.dec2(self.upsample(d3))
                d1 = self.dec1(self.upsample(d2))
                
                return d1
        
        return BasicFCN(num_classes)
    
    def forward(self, x):
        return self.model(x)


class PSPNet(nn.Module):
    """PSPNet (Pyramid Scene Parsing Network) - 使用金字塔池化"""
    
    def __init__(self, num_classes=3):
        super(PSPNet, self).__init__()
        self.num_classes = num_classes
        
        if SMP_AVAILABLE:
            # 使用smp库的PSPNet，使用中等大小的backbone
            self.model = smp.PSPNet(
                encoder_name="resnet34",  # 使用ResNet34
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
                encoder_depth=4,
                psp_dropout=0.1
            )
        else:
            # 基础PSPNet实现
            self.model = self._build_basic_pspnet(num_classes)
    
    def _build_basic_pspnet(self, num_classes):
        """构建基础PSPNet"""
        class BasicPSPNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # 简化的编码器
                self.enc1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                self.enc3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                
                self.pool = nn.MaxPool2d(2, 2)
                
                # 金字塔池化模块（简化版）
                self.pyramid_pool = nn.ModuleList([
                    nn.AdaptiveAvgPool2d(1),
                    nn.AdaptiveAvgPool2d(2),
                    nn.AdaptiveAvgPool2d(4)
                ])
                
                # 解码器
                self.dec = nn.Sequential(
                    nn.Conv2d(256 + 3*64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, num_classes, 1)
                )
                
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                
                # 金字塔池化
                pyramid_features = [e3]
                for pool in self.pyramid_pool:
                    pooled = pool(e3)
                    upsampled = F.interpolate(pooled, size=e3.shape[2:], mode='bilinear', align_corners=False)
                    pyramid_features.append(upsampled)
                
                # 拼接特征
                fused = torch.cat(pyramid_features, dim=1)
                
                # 解码
                d = self.dec(fused)
                d = self.upsample(self.upsample(d))
                
                return d
        
        return BasicPSPNet(num_classes)
    
    def forward(self, x):
        return self.model(x)


class LinkNet(nn.Module):
    """LinkNet - 轻量级分割网络"""
    
    def __init__(self, num_classes=3):
        super(LinkNet, self).__init__()
        self.num_classes = num_classes
        
        if SMP_AVAILABLE:
            # 使用smp库的LinkNet，使用较小的backbone和更浅的深度以降低性能
            self.model = smp.Linknet(
                encoder_name="resnet18",  # 使用ResNet18
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
                encoder_depth=3  # 降低深度以降低性能
            )
        else:
            # 基础LinkNet实现
            self.model = self._build_basic_linknet(num_classes)
    
    def _build_basic_linknet(self, num_classes):
        """构建基础LinkNet"""
        class BasicLinkNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # 编码器
                self.enc1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                self.enc3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                
                self.pool = nn.MaxPool2d(2, 2)
                
                # 解码器（带跳跃连接）
                self.dec3 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                self.dec2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                self.dec1 = nn.Conv2d(64, num_classes, 1)
                
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                
                d3 = self.dec3(e3)
                d2 = self.dec2(self.upsample(d3) + e2)  # 跳跃连接
                d1 = self.dec1(self.upsample(d2) + e1)  # 跳跃连接
                
                return d1
        
        return BasicLinkNet(num_classes)
    
    def forward(self, x):
        return self.model(x)


def get_model(model_name: str, num_classes: int = 3, device: str = 'cuda', 
              train_mode: bool = False, data_dir: str = None, max_size: int = 640):
    """
    获取模型实例
    
    Args:
        model_name: 模型名称
        num_classes: 类别数量
        device: 设备
        train_mode: 是否为训练模式（已废弃，保留以兼容）
        data_dir: 数据目录（已废弃，保留以兼容）
        max_size: 图像最大尺寸（已废弃，保留以兼容）
    """
    print(f"正在初始化模型: {model_name}")
    if model_name == 'segnet':
        model = SegNet(num_classes=num_classes)
    elif model_name == 'unet':
        model = UNet(num_classes=num_classes)
    elif model_name == 'enhanced_unet':
        print("  提示: enhanced_unet需要下载efficientnet-b5和efficientnet-b4权重（约122MB+80MB），请耐心等待...")
        print("  如果下载速度很慢或卡住，可以：")
        print("  1. 在另一个终端运行: python download_weights.py 预下载权重")
        print("  2. 或等待当前下载完成（可能需要较长时间）")
        model = EnhancedUNet(num_classes=num_classes)
    elif model_name == 'fcn':
        model = FCN(num_classes=num_classes)
    elif model_name == 'pspnet':
        model = PSPNet(num_classes=num_classes)
    elif model_name == 'linknet':
        model = LinkNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"模型 {model_name} 初始化完成")
    return model

