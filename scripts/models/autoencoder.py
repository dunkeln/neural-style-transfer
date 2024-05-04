import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import chain
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s - shape: %(message)s",
    handlers=[logging.StreamHandler()])


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        # freeze encoding weights for pre-trained model
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.encoder = nn.Sequential(*layers)

    def forward(self, img):
        encoded = self.encoder(img)
        logging.debug(tuple(encoded.size()))
        return encoded


class Decoder(nn.Module):
    def __init__(self, encoder: Encoder, constant_dim=False):
        super(Decoder, self).__init__()
        isConv = lambda x: isinstance(x, nn.Conv2d)
        encoder = list(reversed(list(filter(isConv, list(encoder.children())))))

        self.decoder = map(
                lambda layer:
                Decoder.transpose_conv(layer[1], preserve_dim=constant_dim | (layer[0] != len(encoder)-1)),
                enumerate(encoder))
        self.decoder = list(chain.from_iterable(self.decoder))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        logging.debug(tuple(decoded.size()))
        return decoded

    @staticmethod
    def transpose_conv(conv: nn.Conv2d, preserve_dim=True):
        return [
                nn.ConvTranspose2d(
                    in_channels=conv.out_channels,
                    out_channels=conv.in_channels,
                    kernel_size=3 if preserve_dim else 4,
                    padding=1,
                    stride=1 if preserve_dim else 2,
                ), nn.ReLU()
                ]


class AutoEncoder(nn.Module):
    def __init__(self, base_model: nn.Sequential):
        """
            Universal Style Transfer Model for VGG-19 net

            dimension changes with encoding layers:
                (224, 224) -> (112, 112) -> (56, 56) -> (28, 28) -> (14, 14)
            dimension changes with decoding layers:
                (14, 14) -> (28, 28) -> (56, 56) -> (112, 112) -> (224, 224)

            Params:
                base_model(torch.nn.Sequential): Convolutional layer of VGG19
        """
        super(AutoEncoder, self).__init__()

        # debug: all encoders chunked down
        self.encoder = OrderedDict()
        encoder = []
        ref = 0
        for layer in list(base_model.children()):
            if isinstance(layer, nn.MaxPool2d):
                self.encoder.update({
                    f'layer_{(ref := ref+1)}': Encoder(encoder)
                })
                encoder = []
            encoder.append(layer)

        self.encoder = nn.Sequential(self.encoder)

        self.decoder = OrderedDict()
        # mirror encoder
        for i in reversed(range(len(self.encoder))):
            layer = f"layer_{i+1}"
            encoder = self.encoder.get_submodule(layer).encoder
            self.decoder.update({
                layer: Decoder(encoder, constant_dim=(i + 1 == 5))
                })

        self.decoder = nn.Sequential(self.decoder)

    def forward(self, img):
        encoded = self.encoder(img)
        decoded = self.decoder(encoded)
        return decoded