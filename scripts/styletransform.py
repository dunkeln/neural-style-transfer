import torch
import torch.nn as nn
from collections import OrderedDict
import logging

from .autoencoder import AutoEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - shape: %(message)s",
    handlers=[logging.StreamHandler()])

# write style transform model here
class StyleTransformModel(AutoEncoder):
    def __init__(self, base_model):
        super(StyleTransformModel, self).__init__(base_model)

    @staticmethod
    def WCT(img, style_img):
        """
        Whitening and Coloring transforms

        """
    def forward(self, img, style_img):
        
        pass