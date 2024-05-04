import torch
import torch.nn as nn
from collections import OrderedDict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - shape: %(message)s",
    handlers=[logging.StreamHandler()])

# write style transform model here