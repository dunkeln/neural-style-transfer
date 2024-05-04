import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from collections import OrderedDict


class StyleTransfer(nn.Module):
    def __init__(self, base_model: nn.Sequential):
        super(StyleTransfer, self).__init__()
        # freeze parameters in the "pre-trained" model
        for param in base_model.parameters():
            param.requires_grad = False

        layers = list(base_model.children())
        encoder = []
        self.autoencoder = nn.Sequential()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        for layer in layers:
            if isinstance(layer, nn.MaxPool2d):
                self.autoencoder.append(nn.Sequential(OrderedDict([
                    ('Encoder', nn.Sequential(*encoder))
                    ])))
                encoder = []
            encoder.append(layer)

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
    # img = torch.randn(1, 3, 224, 224)
    # encoded = model(img)
    # print(encoded.size())
    print(model)
