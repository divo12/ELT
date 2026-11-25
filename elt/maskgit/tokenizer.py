import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1024, 256)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = z.shape[0]
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, 256)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * z_flat @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1).reshape(b, 16, 16)
        quantized = self.embedding(indices).permute(0, 3, 1, 2)
        return indices, quantized


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        if self.nin_shortcut is not None:
            shortcut = self.nin_shortcut(shortcut)
        return x + shortcut


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.block = nn.ModuleList(
            [ResBlock(in_channels, out_channels), ResBlock(out_channels, out_channels)]
        )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class VQGANEncoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.down = nn.ModuleList(
            [
                DownStage(128, 128, downsample=True),
                DownStage(128, 128, downsample=True),
                DownStage(128, 256, downsample=True),
                DownStage(256, 256, downsample=True),
                DownStage(256, 512, downsample=False),
            ]
        )
        self.mid = nn.ModuleList([ResBlock(512, 512), ResBlock(512, 512)])
        self.norm_out = nn.GroupNorm(32, 512)
        self.conv_out = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for down in self.down:
            x = down(x)
        for res_block in self.mid:
            x = res_block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class UpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True):
        super().__init__()
        self.block = nn.ModuleList(
            [ResBlock(in_channels, out_channels), ResBlock(out_channels, out_channels)]
        )
        self.upsample = upsample
        self.upsample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if upsample else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        if self.upsample and self.upsample_conv is not None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.upsample_conv(x)
        return x


class VQGANDecoder(nn.Module):
    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.conv_in = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.mid = nn.ModuleList([ResBlock(512, 512), ResBlock(512, 512)])
        self.up = nn.ModuleList(
            [
                UpStage(128, 128, upsample=False),
                UpStage(256, 128, upsample=True),
                UpStage(256, 256, upsample=True),
                UpStage(512, 256, upsample=True),
                UpStage(512, 512, upsample=True),
            ]
        )
        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for res_block in self.mid:
            x = res_block(x)
        for stage in reversed(self.up):
            x = stage(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class VQGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VQGANEncoder()
        self.quantize = VectorQuantizer()
        self.decoder = VQGANDecoder()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(self.encoder(x))

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices, quantized = self.encode(x)
        return self.decode(quantized), indices


def load_tokenizer(
    device: torch.device, ckpt_path: str = "ckpts/maskgit-vqgan-imagenet-f16-256.bin"
) -> VQGAN:
    vqgan = VQGAN()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    vqgan.load_state_dict(state_dict, strict=False)
    vqgan.to(device)
    vqgan.eval()
    for param in vqgan.parameters():
        param.requires_grad = False
    print(f"tokenizer loaded ({sum(p.numel() for p in vqgan.parameters()):,} params)")
    return vqgan


def encode(vqgan: VQGAN, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        indices, _ = vqgan.encode(images)
        return indices.reshape(images.shape[0], -1)


def decode(vqgan: VQGAN, indices: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        batch = indices.shape[0]
        indices_2d = indices.clamp(0, 1023).reshape(batch, 16, 16)
        quantized = vqgan.quantize.embedding(indices_2d).permute(0, 3, 1, 2)
        return vqgan.decode(quantized)
