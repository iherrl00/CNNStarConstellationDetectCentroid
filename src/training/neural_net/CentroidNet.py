import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .mobile_unet import get_inference_time


class Block(nn.Module):
    """Bloque convolucional con dos capas"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(c_out)
        self.conv_2 = nn.Conv2d(
            c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # Primera capa
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        # Segunda capa
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    """Codificador con bloques y pooling"""

    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1]) for i in range(len(chs)-2)])
        self.last_block = Block(chs[-2], chs[-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        encoder_features = []

        # Procesar bloques con pooling
        for block in self.enc_blocks:
            x = block(x)
            encoder_features.append(x)
            x = self.pool(x)

        # Último bloque sin pooling
        x = self.last_block(x)

        return x, encoder_features


class Decoder(nn.Module):
    """Decodificador con upsampling y skip connections"""

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2, stride=2, bias=True)
                                      for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            # Upsampling
            x = self.upconvs[i](x)
            x = F.interpolate(x, size=(encoder_features[i].size()[2], encoder_features[i].size()[3]),
                              mode='bilinear', align_corners=True)

            # Skip connection
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)

        return x


class CentroidNet(nn.Module):
    """Red U-Net para detección de centroides"""

    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.conv_last = nn.Conv2d(
            dec_chs[-1], num_class, kernel_size=1, bias=True)

    def forward(self, x):
        # Codificar
        x, encoder_features = self.encoder(x)

        # Decodificar con features invertidos
        out = self.decoder(x, encoder_features[::-1])
        out = self.conv_last(out)

        return out


if __name__ == "__main__":
    # Configurar dispositivo y modelo
    device = torch.device("cuda:0")
    model = CentroidNet(enc_chs=(1, 64, 128, 256, 512, 1024),
                        dec_chs=(1024, 512, 256, 128, 64),
                        num_class=2).to(device)
    input = torch.randn(1, 1, 480, 640).to(device)

    # Medir tiempo de inferencia
    inference_time = get_inference_time(model, input)
    print(f'Tiempo de inferencia = {inference_time/1000} s')

    # Obtener tamaño de salida
    predict = model(input)
    predict = predict.detach().cpu()
    print(f'Tamaño de salida = {predict.size()}')
    print(f'Tamaño del mapa seg/dist = {predict[0][0].size()}')

    # Contar parámetros totales
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Parámetros totales = {pytorch_total_params / 1000000} millones')

    # Calcular MACs y parámetros
    macs, params = profile(model, inputs=(input,))
    print(f'MACs = {macs/10**9}G')
    print(f'Params = {params/1000**2}M')
