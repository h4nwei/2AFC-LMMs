from pathlib import Path
from importlib import import_module
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')
        module = import_module('LMMs.' + args.model_name)
        self.model = module.make_model()


    def forward(self, imgPath1, imgPath2):
        return self.model.run(imgPath1, imgPath2)
