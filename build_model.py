import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from typing import Iterable, Optional, Set


def build_model(args):
    model_name = args.model_name.lower()
    backbone_name = getattr(args, "backbone_name", "mobilenet_v2")

    if model_name == "convstack":
        return SpectrumConvStack(
            backbone_name=backbone_name,
            output_dim=args.output_dim,
            pretrained=args.pretrained,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_mcsr=args.use_mcsr,
        )

    raise ValueError(f"Unknown model_name: {args.model_name}")


def savgol_kernel(window_length: int, polyorder: int) -> torch.Tensor:
    from scipy.linalg import pinv
    import numpy as np

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")

    half_window = (window_length - 1) // 2
    x = np.arange(-half_window, half_window + 1)
    A = np.vander(x, polyorder + 1)

    ATA_inv = pinv(A.T @ A)
    coeffs = ATA_inv @ A.T
    kernel = coeffs[0]

    return torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1)


class SavitzkyGolayLayer(nn.Module):
    def __init__(self, window_length: int, polyorder: int):
        super().__init__()
        kernel = savgol_kernel(window_length, polyorder)
        self.register_buffer("kernel", kernel)
        self.window_length = window_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.kernel, padding=self.window_length // 2)


def _replace_first_conv_with_grayscale(module: nn.Module):
    """
    Replace the first Conv2d found in a model with a 1-channel version.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
            )

            with torch.no_grad():
                if child.weight.shape[1] == 3:
                    new_conv.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                else:
                    new_conv.weight.copy_(child.weight)

                if child.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(child.bias)

            setattr(module, name, new_conv)
            return True

        replaced = _replace_first_conv_with_grayscale(child)
        if replaced:
            return True

    return False


def _build_torchvision_backbone(backbone_name: str, pretrained: bool, dropout: float, image_size: int,):
    backbone_name = backbone_name.lower()

    if backbone_name == "tiny_cnn":
        model = TinyCNNBackboneNet(
            in_channels=1,
            feature_dim=256,
            dropout=dropout,
        )
        feature_dim = 256
        return model, feature_dim

    if backbone_name == "mlp_baseline":
        model = ImageMLPBackboneNet(
            image_size=image_size,
            in_channels=1,
            hidden_dims=[1024, 512],
            feature_dim=256,
            dropout=dropout,
        )
        feature_dim = 256
        return model, feature_dim

    # MobileNetV2
    if backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        _replace_first_conv_with_grayscale(model.features)

        in_features = model.last_channel
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, in_features),
        )
        feature_dim = in_features
        return model, feature_dim

    # VGG11 BN
    if backbone_name == "vgg11_bn":
        weights = models.VGG11_BN_Weights.DEFAULT if pretrained else None
        model = models.vgg11_bn(weights=weights)
        _replace_first_conv_with_grayscale(model.features)

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, in_features),
        )
        feature_dim = in_features
        return model, feature_dim

    # VGG16 BN
    if backbone_name == "vgg16_bn":
        weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
        model = models.vgg16_bn(weights=weights)
        _replace_first_conv_with_grayscale(model.features)

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, in_features),
        )
        feature_dim = in_features
        return model, feature_dim

    # ResNet18
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, in_features),
        )
        feature_dim = in_features
        return model, feature_dim

    # ConvNeXt Tiny
    if backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        _replace_first_conv_with_grayscale(model.features)

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, in_features),
        )
        feature_dim = in_features
        return model, feature_dim

    # ViT-B/16
    if backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)

        old_conv = model.conv_proj
        model.conv_proj = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            model.conv_proj.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None and model.conv_proj.bias is not None:
                model.conv_proj.bias.copy_(old_conv.bias)

        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, in_features)
        feature_dim = in_features
        return model, feature_dim

    raise ValueError(
        f"Unsupported backbone_name: {backbone_name}. "
        f"Supported: mobilenet_v2, vgg11_bn, vgg16_bn, resnet18, "
        f"efficientnet_b0, convnext_tiny, vit_b_16"
    )


class SpectrumBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenet_v2",
        image_size: int = 64,
        output_dim: int = 102,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.base_model, feature_dim = _build_torchvision_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            dropout=dropout,
            image_size=image_size,
        )

        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.fc(x)
        x = self.out_proj(x)
        x = x.unsqueeze(1)
        return x


class SpectrumConvStack(SpectrumBackbone):
    """
    Backwards-compatible replacement for mobilenet_v2_spectrum_convstack,
    but now supports multiple backbones.
    """
    def __init__(
        self,
        backbone_name: str = "mobilenet_v2",
        image_size: int = 64,
        output_dim: int = 102,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_savgol: bool = False,
        savgol_pos: str = "after_mcsr",
        savgol_window: int = 11,
        savgol_polyorder: int = 2,
        use_mcsr: bool = True,
    ):
        super().__init__(
            backbone_name=backbone_name,
            image_size=image_size,
            output_dim=output_dim,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        assert savgol_pos in ["before_mcsr", "after_mcsr"], (
            "savgol_pos must be 'before_mcsr' or 'after_mcsr'"
        )

        self.use_mcsr = use_mcsr
        self.use_savgol = use_savgol
        self.savgol_pos = savgol_pos

        if self.use_savgol:
            self.savgol_layer = SavitzkyGolayLayer(
                window_length=savgol_window,
                polyorder=savgol_polyorder,
            )

        self.conv1d1 = nn.Conv1d(1, 5, kernel_size=5, padding=2)
        self.conv1d = nn.Conv1d(5, 1, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        if self.use_savgol and self.savgol_pos == "before_mcsr":
            x = self.savgol_layer(x)

        if self.use_mcsr:
            x = self.conv1d1(x)
            x = self.conv1d(x)

        if self.use_savgol and self.savgol_pos == "after_mcsr":
            x = self.savgol_layer(x)

        x = x.squeeze(1)
        return x


##########################################

class TinyCNNBackboneNet(nn.Module):
    """
    Small CNN backbone for ablation.
    Returns a feature vector.
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.proj(x)
        return x


#########################################

class ImageMLPBackboneNet(nn.Module):
    """
    Image MLP backbone for ablation.
    Flattens the image and returns a feature vector.
    """
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 1,
        hidden_dims=None,
        feature_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        input_dim = in_channels * image_size * image_size

        layers = [nn.Flatten()]
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

