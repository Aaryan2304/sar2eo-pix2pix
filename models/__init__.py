import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    """Single U-Net encoder/decoder block."""

    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, dropout_rate=0.5):
        super().__init__()

        if down:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]

        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net generator. 256x256 in, 256x256 out. 8 encoder levels, skip-connected."""

    def __init__(self, in_channels=1, out_channels=3, base_filters=64, num_layers=8,
                 use_dropout=True, dropout_rate=0.5):
        super().__init__()

        enc_channels = [in_channels]
        for i in range(num_layers):
            enc_channels.append(min(base_filters * (2 ** i), 512))

        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            self.encoders.append(UNetBlock(enc_channels[i], enc_channels[i + 1], down=True))

        # Separate upsamplers from post-concat convs.
        # Concat only makes sense AFTER upsampling (hit a size mismatch bug otherwise).
        self.upsamplers = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()

        for i in range(num_layers - 2, -1, -1):
            in_ch = enc_channels[i + 1]
            concat_ch = in_ch + enc_channels[i + 1]
            # Outer level outputs 64ch so final layer maps 64→3 instead of 1→3
            out_ch = enc_channels[i] if i > 0 else enc_channels[1]

            use_drop = use_dropout and (i >= num_layers - 3)

            self.upsamplers.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=False)
            )

            layers = [
                nn.Conv2d(concat_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if use_drop:
                layers.append(nn.Dropout(dropout_rate))
            self.decoder_convs.append(nn.Sequential(*layers))

        self.final = nn.Sequential(
            nn.ConvTranspose2d(enc_channels[1], out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self._num_layers = num_layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        for i in range(len(self.upsamplers)):
            x = self.upsamplers[i](x)
            x = torch.cat([x, skips[self._num_layers - 2 - i]], dim=1)
            x = self.decoder_convs[i](x)

        return self.final(x)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN. Takes SAR+EO (4ch), outputs 30x30 predictions.
    No sigmoid — use BCEWithLogitsLoss directly."""

    def __init__(self, in_channels=4, base_filters=64, num_layers=4):
        super().__init__()

        layers = []
        ch = base_filters

        # First layer skips BN (Pix2Pix convention)
        layers.append(nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            ch *= 2

        layers.extend([
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        ch *= 2

        layers.append(nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, sar, eo):
        return self.model(torch.cat([sar, eo], dim=1))
