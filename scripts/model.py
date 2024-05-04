import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from itertools import chain
import logging

logging.basicConfig(
    level=logging.INFO,
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
        logging.info(tuple(encoded.size()))
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
        logging.info(tuple(decoded.size()))
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


class StyleTransfer(nn.Module):
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
        super(StyleTransfer, self).__init__()

        # INFO: all encoders chunked down
        self.autoencoder = OrderedDict()
        encoder = []
        ref = 0
        for layer in list(base_model.children()):
            if isinstance(layer, nn.MaxPool2d):
                self.autoencoder.update({
                    f'layer_{(ref := ref+1)}': Encoder(encoder)
                })
                encoder = []
            encoder.append(layer)

        self.autoencoder = nn.Sequential(self.autoencoder)

        self.decoder = OrderedDict()
        # mirror encoder
        for i in reversed(range(len(self.autoencoder))):
            layer = f"layer_{i+1}"
            encoder = self.autoencoder.get_submodule(layer).encoder
            self.decoder.update({
                layer: Decoder(encoder, constant_dim=(i + 1 == 5))
                })

        self.decoder = nn.Sequential(self.decoder)

    def forward(self, img):
        encoded = self.autoencoder(img)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    model = StyleTransfer(
                models
                .vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
                .features
            )
    img = torch.randn(1, 3, 224, 224)
    decoded = model(img)
    print(decoded.shape)
