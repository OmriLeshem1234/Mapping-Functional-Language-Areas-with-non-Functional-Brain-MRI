"""
PyTorch Implementation of AGYnet Multimodal CNN Segmentation Network
- Uses multi-sequence brain MRI data, inspired by:
  Nelkenbaum et al., "Automatic segmentation of white matter tracts using multiple brain MRI sequences,"
  2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI).
- Implements core model architecture, preprocessing pipeline, and training framework in PyTorch.
- Reference: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9098454)
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """Encoder network."""

    def __init__(self, conf, rgb=False, p_dropout=0.5, device="cuda"):
        super().__init__()
        self.device = device
        channels = conf['ynet_ch']
        in_channels = 3 if rgb else 1

        # Layer 1
        self.prelu1 = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, stride=1, padding='same'),
            nn.PReLU()
        )
        self.dropout1 = nn.Sequential(
            nn.Conv3d(channels[0], channels[1], kernel_size=2, stride=2, padding='valid'),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv3d(channels[1], channels[1], kernel_size=3, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[1], channels[1], kernel_size=3, stride=1, padding='same')
        )
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Sequential(
            nn.Conv3d(channels[1], channels[2], kernel_size=2, stride=2, padding='valid'),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv3d(channels[2], channels[2], kernel_size=3, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[2], channels[2], kernel_size=3, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[2], channels[2], kernel_size=3, stride=1, padding='same')
        )
        self.prelu3 = nn.PReLU()
        self.dropout3 = nn.Sequential(
            nn.Conv3d(channels[2], channels[3], kernel_size=2, stride=2, padding='valid'),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=1, padding='same')
        )
        self.prelu4 = nn.PReLU()
        self.dropout4 = nn.Sequential(
            nn.Conv3d(channels[3], channels[4], kernel_size=2, stride=2, padding='valid'),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv3d(channels[4], channels[4], kernel_size=3, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[4], channels[4], kernel_size=3, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[4], channels[4], kernel_size=3, stride=1, padding='same')
        )
        self.prelu5 = nn.PReLU()
        self.dropout5 = nn.Sequential(
            nn.ConvTranspose3d(channels[4], channels[2], kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

    def forward(self, x):
        # Store intermediate activations
        prelu_dict = {}

        # Layer 1
        prelu_dict[1] = self.prelu1(x)
        x = self.dropout1(prelu_dict[1])

        # Layer 2
        residual = self.layer2(x)
        prelu_dict[2] = self.prelu2(residual + x)
        x = self.dropout2(prelu_dict[2])

        # Layer 3
        residual = self.layer3(x)
        prelu_dict[3] = self.prelu3(residual + x)
        x = self.dropout3(prelu_dict[3])

        # Layer 4
        residual = self.layer4(x)
        prelu_dict[4] = self.prelu4(residual + x)
        x = self.dropout4(prelu_dict[4])

        # Layer 5
        residual = self.layer5(x)
        prelu_dict[5] = self.prelu5(residual + x)
        x = self.dropout5(prelu_dict[5])

        return prelu_dict, x


class AttentionGate(nn.Module):
    """Attention gate for 3D inputs with skip connections."""

    def __init__(self, in_channels, in_shape, return_mask=False, mri_dims=(160, 192, 160), device="cuda"):
        super().__init__()
        self.in_shape = in_shape
        self.return_mask = return_mask
        self.mri_dims = mri_dims
        self.in_channels = in_channels
        self.device = device

        # Calculate intermediate channel dimensions
        inter_channels = max(in_channels // 2, 1)

        # Gating operations
        self.Wx = nn.Conv3d(in_channels, inter_channels, 2, stride=2, padding='valid')
        self.Wg = nn.Conv3d(in_channels * 2, inter_channels, 1, stride=1, padding='valid', bias=True)

        # Attention computation path
        self.path = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding='valid', bias=True),
            nn.Sigmoid()
        )

        # Transformation after attention
        self.Wy = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding='valid', bias=True)

    def forward(self, x, xg, g):
        # Compute attention weights
        theta_g = self.Wx(xg)
        phi_g = self.Wg(g)

        attention_sum = theta_g + phi_g
        attention_mask = self.path(attention_sum)

        # Free memory
        del theta_g, phi_g, attention_sum
        torch.cuda.empty_cache()

        # Upsample attention mask to match input size
        upsampled_mask = nn.functional.interpolate(
            attention_mask,
            size=self.in_shape,
            mode='trilinear'
        )

        # Return mask or apply attention
        if self.return_mask:
            return upsampled_mask
        else:
            attended_features = upsampled_mask * x
            return self.Wy(attended_features) + x


class Decoder(nn.Module):
    """Decoder network with attention gates."""

    def __init__(self, conf, mri_dims=(160, 192, 160), use_final_activation=True, p_dropout=0.5, device="cuda"):
        super().__init__()
        self.mri_dims = mri_dims
        self.device = device
        k = 3
        channels = conf['ynet_ch']
        self.use_final_activation = use_final_activation

        # Create attention gates for each layer
        # Layer 4
        self.ag4_t1 = AttentionGate(
            in_channels=channels[3],
            in_shape=(mri_dims[0] // 8, mri_dims[1] // 8, mri_dims[2] // 8),
            device=device
        ).to(device)

        self.ag4_rgb = AttentionGate(
            in_channels=channels[3],
            in_shape=(mri_dims[0] // 8, mri_dims[1] // 8, mri_dims[2] // 8),
            device=device
        ).to(device)

        self.layer4a = nn.Sequential(
            nn.Conv3d(2 * channels[3] + 2 * channels[2], channels[3], k, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[3], channels[3], k, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[3], channels[3], k, stride=1, padding='same')
        )
        self.layer4b = nn.PReLU()
        self.layer4c = nn.Sequential(
            nn.ConvTranspose3d(channels[3], channels[3], 2, stride=2, padding=0),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        # Layer 3
        self.ag3_t1 = AttentionGate(
            in_channels=channels[2],
            in_shape=(mri_dims[0] // 4, mri_dims[1] // 4, mri_dims[2] // 4),
            device=device
        ).to(device)

        self.ag3_rgb = AttentionGate(
            in_channels=channels[2],
            in_shape=(mri_dims[0] // 4, mri_dims[1] // 4, mri_dims[2] // 4),
            device=device
        ).to(device)

        self.layer3a = nn.Sequential(
            nn.Conv3d(channels[4], channels[3], k, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[3], channels[3], k, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[3], channels[3], k, stride=1, padding='same')
        )
        self.layer3b = nn.PReLU()
        self.layer3c = nn.Sequential(
            nn.ConvTranspose3d(channels[3], channels[2], 2, stride=2, padding=0),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        # Layer 2
        self.ag2_t1 = AttentionGate(
            in_channels=channels[1],
            in_shape=(mri_dims[0] // 2, mri_dims[1] // 2, mri_dims[2] // 2),
            device=device
        ).to(device)

        self.ag2_rgb = AttentionGate(
            in_channels=channels[1],
            in_shape=(mri_dims[0] // 2, mri_dims[1] // 2, mri_dims[2] // 2),
            device=device
        ).to(device)

        self.layer2a = nn.Sequential(
            nn.Conv3d(channels[3], channels[2], k, stride=1, padding='same'),
            nn.PReLU(),
            nn.Conv3d(channels[2], channels[2], k, stride=1, padding='same')
        )
        self.layer2b = nn.PReLU()
        self.layer2c = nn.Sequential(
            nn.ConvTranspose3d(channels[2], channels[1], 2, stride=2, padding=0),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        # Layer 1
        self.ag1_t1 = AttentionGate(
            in_channels=channels[0],
            in_shape=mri_dims,
            device=device
        ).to(device)

        self.ag1_rgb = AttentionGate(
            in_channels=channels[0],
            in_shape=mri_dims,
            device=device
        ).to(device)

        self.layer1a = nn.Sequential(
            nn.Conv3d(channels[2], channels[1], k, stride=1, padding='same')
        )
        self.layer1b = nn.PReLU()
        self.layer1c = nn.Conv3d(channels[1], conf['class_num'] + 1, 1, stride=1, padding='same')

        # Final activation
        if conf["use_sigmoid"]:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, pre_dict_t1, dropout_t1, pre_dict_rgb, dropout_rgb):

        # Layer 4
        merge_t1_rgb = torch.cat((dropout_t1, dropout_rgb), dim=1)
        prelu_4_t1_fused = self.ag4_t1(pre_dict_t1[4], pre_dict_rgb[4], pre_dict_rgb[5])
        prelu_4_rgb_fused = self.ag4_rgb(pre_dict_rgb[4], pre_dict_t1[4], pre_dict_t1[5])

        merge_with_fusion = torch.cat((prelu_4_t1_fused, prelu_4_rgb_fused, dropout_t1, dropout_rgb), dim=1)
        dropout_4_rgbt1 = self.layer4c(
            self.layer4b(merge_t1_rgb + self.layer4a(merge_with_fusion))
        )

        # Memory cleanup
        del merge_with_fusion, merge_t1_rgb, dropout_t1, dropout_rgb
        del pre_dict_t1[5], pre_dict_rgb[5], pre_dict_t1[4], pre_dict_rgb[4]
        torch.cuda.empty_cache()

        # Layer 3
        prelu_3_t1_fused = self.ag3_t1(pre_dict_t1[3], pre_dict_rgb[3], prelu_4_rgb_fused)
        prelu_3_rgb_fused = self.ag3_rgb(pre_dict_rgb[3], pre_dict_t1[3], prelu_4_t1_fused)
        merge_3 = torch.cat((dropout_4_rgbt1, prelu_3_t1_fused, prelu_3_rgb_fused), dim=1)
        dropout_3_rgbt1 = self.layer3c(
            self.layer3b(dropout_4_rgbt1 + self.layer3a(merge_3))
        )

        # Memory cleanup
        del dropout_4_rgbt1, merge_3, prelu_4_t1_fused, prelu_4_rgb_fused
        del pre_dict_t1[3], pre_dict_rgb[3]
        torch.cuda.empty_cache()

        # Layer 2
        prelu_2_t1_fused = self.ag2_t1(pre_dict_t1[2], pre_dict_rgb[2], prelu_3_rgb_fused)
        prelu_2_rgb_fused = self.ag2_rgb(pre_dict_rgb[2], pre_dict_t1[2], prelu_3_t1_fused)
        merge_2 = torch.cat((dropout_3_rgbt1, prelu_2_t1_fused, prelu_2_rgb_fused), dim=1)
        dropout_2_rgbt1 = self.layer2c(
            self.layer2b(dropout_3_rgbt1 + self.layer2a(merge_2))
        )

        # Memory cleanup
        del dropout_3_rgbt1, merge_2, prelu_3_t1_fused, prelu_3_rgb_fused
        del pre_dict_t1[2], pre_dict_rgb[2]
        torch.cuda.empty_cache()

        # Layer 1
        prelu_1_t1_fused = self.ag1_t1(pre_dict_t1[1], pre_dict_rgb[1], prelu_2_rgb_fused)
        prelu_1_rgb_fused = self.ag1_rgb(pre_dict_rgb[1], pre_dict_t1[1], prelu_2_t1_fused)
        merge_1 = torch.cat((dropout_2_rgbt1, prelu_1_t1_fused, prelu_1_rgb_fused), dim=1)

        # Final processing and activation
        pre_output = self.layer1c(
            self.layer1b(dropout_2_rgbt1 + self.layer1a(merge_1))
        )

        if self.use_final_activation:
            output = self.final_activation(pre_output)
        else:
            output = pre_output

        # Memory cleanup
        del dropout_2_rgbt1, merge_1, prelu_2_t1_fused, prelu_2_rgb_fused
        del pre_dict_t1[1], pre_dict_rgb[1], prelu_1_t1_fused, prelu_1_rgb_fused
        torch.cuda.empty_cache()

        return output


class AGYnet(nn.Module):
    """Complete AGYnet with dual path and attention gates."""

    def __init__(self, out_channels: int, ynet_ch: list = [20, 40, 80, 160, 320],
                 mri_dims: tuple = (160, 192, 160), p_dropout: float = 0.5, device="cuda"):
        super().__init__()

        # Ensure MRI dimensions are divisible by 16
        if any(dim % 16 != 0 for dim in mri_dims):
            raise ValueError(f"Each dimension of MRI input must be divisible by 16, got {mri_dims}")

        self.device = device

        self.ynet_ch = ynet_ch
        self.out_channels = out_channels
        self.mri_dims = mri_dims

        # Configure network
        self.use_sigmoid = (out_channels == 1)
        self.conf = self._init_config()

        # Create network components
        self.t1_path = Encoder(conf=self.conf, rgb=False, p_dropout=p_dropout, device=device).to(device)
        self.rgb_path = Encoder(conf=self.conf, rgb=True, p_dropout=p_dropout, device=device).to(device)
        self.decoder = Decoder(
            conf=self.conf,
            mri_dims=self.mri_dims,
            use_final_activation=True,  # Always using a final activation
            p_dropout=p_dropout,
            device=device
        ).to(device)

    def _init_config(self):
        """Initialize configuration dictionary."""
        return {
            'ynet_ch': self.ynet_ch,
            'class_num': self.out_channels - 1,  # background not included
            'use_sigmoid': self.use_sigmoid
        }

    def forward(self, t1, rgb):
        """Forward pass through the network."""
        if list(t1.shape[2:]) != self.mri_dims or list(rgb.shape[2:]) != self.mri_dims:
            raise ValueError(f"Input tensor shape must match MRI dims {self.mri_dims}, "
                             f"got {t1.shape[2:]} for t1 and {rgb.shape[2:]} for rgb")

        pre_dict_t1, dropout_t1 = self.t1_path(t1)
        pre_dict_rgb, dropout_rgb = self.rgb_path(rgb)
        output = self.decoder(pre_dict_t1, dropout_t1, pre_dict_rgb, dropout_rgb)
        return output


class AGYNetConcatenatedInput(AGYnet):
    """AGYnet variant that accepts concatenated input."""

    def __init__(self, out_channels: int, ynet_ch: list = [20, 40, 80, 160, 320],
                 mri_dims: tuple = (160, 192, 160), p_dropout: float = 0.5, device="cuda"):
        super().__init__(out_channels, ynet_ch, mri_dims, p_dropout, device)
        self.name = "AGYNetConcatenatedInput"

    def forward(self, inputs):
        """Forward pass with concatenated inputs."""
        t1 = torch.unsqueeze(inputs[:, 0, ...], 1)
        rgb = inputs[:, 1:, ...]
        return super().forward(t1, rgb)
