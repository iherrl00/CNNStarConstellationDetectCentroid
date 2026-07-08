"""
    Basado en "Mobile-Unet: An efficient convolutional neural network for fabric defect detection"
"""

import torch
import torch.nn as nn
import numpy as np

from thop import profile


class InvertedResidualBlock(nn.Module):
    """
    Bloque residual invertido de MobileNetV2.
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=6):
        super(InvertedResidualBlock, self).__init__()
        # Validar el stride.
        assert stride in [1, 2]
        # Hay salto si el stride es 1.
        self.use_skip_connection = True if stride == 1 else False

        # Factor de expansión.
        ex_c = int(in_c * expansion_factor)
        self.conv = nn.Sequential(
            # Conv 1x1.
            nn.Conv2d(in_c, ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ex_c),
            nn.ReLU6(inplace=True),
            # Conv depthwise.
            nn.Conv2d(ex_c, ex_c, 3, stride, 1, groups=ex_c, bias=False),
            nn.BatchNorm2d(ex_c),
            nn.ReLU6(inplace=True),
            # Conv 1x1 final.
            nn.Conv2d(ex_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.use_conv1x1 = False if in_c == out_c else True
        self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.use_skip_connection:
            out = self.conv(x)
            if self.use_conv1x1:
                x = self.conv1x1(x)
            return x+out
        else:
            return self.conv(x)


class MobileUNet(nn.Module):
    """
    U-Net modificado con bloques residuales invertidos y convolución separable.
    """

    def __init__(self, ch_in, ch_out):
        super(MobileUNet, self).__init__()

        # Rama de encoder.
        self.D1 = nn.Sequential(*([nn.Conv2d(ch_in, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(
            16), nn.ReLU6(inplace=True)]+list(self.irb_bottleneck(16, 16, 1, 1, 1))))
        self.D2 = self.irb_bottleneck(16, 24, 2, 2, 6)
        self.D3 = self.irb_bottleneck(24, 32, 3, 2, 6)
        self.D4 = nn.Sequential(
            *(list(self.irb_bottleneck(32, 96, 4, 2, 6)) + list(self.irb_bottleneck(96, 96, 3, 1, 6))))
        self.D5 = nn.Sequential(*(list(self.irb_bottleneck(96, 128, 3, 2, 6)) + list(self.irb_bottleneck(128, 128, 1, 1, 6)) + list(
            self.irb_bottleneck(128, 128, 1, 1, 6)) + [nn.Conv2d(128, 1280, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(1280)]))
        # Rama de decoder.
        self.ConvTranspose1 = nn.Sequential(
            *[nn.ConvTranspose2d(1280, 96, 4, 2, 1), nn.BatchNorm2d(96)])
        self.InvertedResidual1 = self.irb_bottleneck(96+96, 96, 1, 1, 6, True)
        self.ConvTranspose2 = nn.Sequential(
            *[nn.ConvTranspose2d(96, 32, 4, 2, 1), nn.BatchNorm2d(32)])
        self.InvertedResidual2 = self.irb_bottleneck(32+32, 32, 1, 1, 6, True)
        self.ConvTranspose3 = nn.Sequential(
            *[nn.ConvTranspose2d(32, 24, 4, 2, 1), nn.BatchNorm2d(24)])
        self.InvertedResidual3 = self.irb_bottleneck(24+24, 24, 1, 1, 6, True)
        self.ConvTranspose4 = nn.Sequential(
            *[nn.ConvTranspose2d(24, 16, 4, 2, 1), nn.BatchNorm2d(16)])
        self.InvertedResidual4 = self.irb_bottleneck(16+16, 16, 1, 1, 6, True)
        self.ConvTranspose5 = nn.ConvTranspose2d(16, ch_out, 4, 2, 1)

    def irb_bottleneck(self, in_c, out_c, n, s, t, d=False):
        """
            Crea una serie de bloques residuales invertidos.
            @param in_c
            @param out_c
            @param n: número de repeticiones
            @param s: stride
        """
        convs = []
        if d:
            xx = InvertedResidualBlock(in_c, out_c, s, t)
            convs.append(xx)
        else:
            xx = InvertedResidualBlock(in_c, out_c, s, t)
            convs.append(xx)
            if n > 1:
                for i in range(1, n):
                    xx = InvertedResidualBlock(out_c, out_c, 1, t)
                    convs.append(xx)

        conv = nn.Sequential(*convs)
        return conv

    def forward(self, input):
        # Tamaños de referencia para entrada (Batch_Size, 3, 256, 256).
        # Encoder.
        X1 = self.D1(input)
        # print(f'X1={X1.size()}') # (16, 128, 128)
        X2 = self.D2(X1)
        # print(f'X2={X2.size()}') # (24, 64, 64)
        X3 = self.D3(X2)
        # print(f'X3={X3.size()}') # (32, 32, 32)
        X4 = self.D4(X3)
        # print(f'X4={X4.size()}') # (96, 16, 16)
        X5 = self.D5(X4)
        # print(f'X5={X5.size()}') # (1280, 8, 8)

        # Decoder con saltos.
        L1 = self.ConvTranspose1(X5)
        # print(f'L1={L1.size()}') # (96, 16, 16)
        L2 = self.InvertedResidual1(torch.cat([L1, X4], dim=1))
        # print(f'L2={L2.size()}') # (96, 16, 16)
        L3 = self.ConvTranspose2(L2)
        # print(f'L3={L3.size()}') # (32, 32, 32)
        L4 = self.InvertedResidual2(torch.cat([L3, X3], dim=1))
        # print(f'L4={L4.size()}') # (32, 32, 32)
        L5 = self.ConvTranspose3(L4)
        # print(f'L5={L5.size()}') # (24, 64, 64)
        L6 = self.InvertedResidual3(torch.cat([L5, X2], dim=1))
        # print(f'L6={L6.size()}')  # (24, 64, 64)
        L7 = self.ConvTranspose4(L6)
        # print(f'L7={L7.size()}') # (16, 128, 128)
        L8 = self.InvertedResidual4(torch.cat([L7, X1], dim=1))
        # print(f'L8={L8.size()}') # (16, 128, 128)
        output = self.ConvTranspose5(L8)
        # print(output.size()) # (ch_out, 256, 256)

        return output


def get_inference_time(model, dummy_input):
    """
        https://deci.ai/blog/measure-inference-time-deep-neural-networks/
        @param model:
        @param dummy_input:
        @return mean_syn: tiempo de inferencia en milisegundos
    """

    # Inicializar timers.
    # Usar torch.cuda.Event() para medir en GPU.
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))

    # Calentar la GPU.
    # Así evita la latencia del primer uso.
    for _ in range(10):
        _ = model(dummy_input)

    # Medir rendimiento.
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # Esperar sincronía con GPU.
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # Tiempo en ms.
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    return mean_syn


if __name__ == "__main__":
    print("[MobileUNet]")

    device = torch.device("cuda:0")
    dummy_input = torch.rand(1, 1, 480, 640).to(device)
    model = MobileUNet(ch_in=1, ch_out=2).to(device)
    # torch.onnx.export(model, dummy_input, 'mobile_unet.onnx', opset_version=11)

    # Obtener tiempo de inferencia.
    mean_inference_time = get_inference_time(model, dummy_input)
    print(f'inference time = {mean_inference_time/1000} s')

    # Obtener forma de salida.
    predict = model(dummy_input)
    predict = predict.detach().cpu()
    print(f'the output size is {predict.size()}')
    print(f'the predicted seg/dist map size is {predict[0][0].size()}')\


    """
        Revisar el total de parámetros en un modelo PyTorch.
        https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(
        f'the total number of parameters = {pytorch_total_params / 1000000} millions')

    """ 
        Usar thop para parámetros y FLOPs.
        https://github.com/Lyken17/pytorch-OpCounter
    """
    macs, params = profile(model, inputs=(dummy_input,))
    """
        FLOPs: operaciones de punto flotante.
        MACs: operaciones multiply-accumulate.
        1 MACs ≈ 2 FLOPs.
        MFLOPS = 10**6 FLOPS, GFLOPS = 10**9 FLOPS.
    """
    print('MACs = ' + str(macs/10**9) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
