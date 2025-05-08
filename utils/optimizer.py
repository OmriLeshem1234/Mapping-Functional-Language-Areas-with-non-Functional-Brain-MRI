"""
This file contains an optimizer class that holds an optimizer from MONAI.
It checks which arguments the MONAI optimizer accepts and checks mutual arguments from configuration file.
"""

import torch
import inspect


class optimizer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.opt_args = self.args['optimizer_arguments']
        self.optimizer = self.opt_args['optimizer']

    def create_optimizer(self):
        self._create_optimizer_function()
        self._get_input_args_from_optimizer()
        self._get_mutual_args()
        return self.optimizerFunc(**self.optimArgs)

    def _create_optimizer_function(self):
        optimizerFunc = getattr(torch.optim, self.optimizer)
        setattr(self, 'optimizerFunc', optimizerFunc)

    def _get_input_args_from_optimizer(self):
        inputArgs = inspect.getfullargspec(self.optimizerFunc).args
        setattr(self, 'inputArgs', inputArgs)

    def _get_mutual_args(self):
        optimArgs = {k: self.opt_args[k] for k in self.opt_args if k in self.inputArgs}
        optimArgs['params'] = self.model.parameters()
        setattr(self, 'optimArgs', optimArgs)
