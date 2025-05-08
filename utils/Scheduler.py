"""
This file contains a scheduler class that holds a scheduler from pytorch.
It checks which arguments the pytorch scheduler accepts and checks mutual arguments from configuration file.
"""

import torch
import inspect


class Scheduler:
    def __init__(self, args, optimizer):
        self.args = args
        self.optimizer = optimizer
        self.scheduler_args = getattr(self.args, 'scheduler_arguments', None)

    def create_scheduler(self):
        if self.scheduler_args is None:
            return None
        self._create_scheduler_function()
        self._get_input_args_from_optimizer()
        self._get_mutual_args()
        return self.schedulerFunc(**self.schedulerArgs)

    def _create_scheduler_function(self):
        schedulerFunc = getattr(torch.optim.lr_scheduler, self.scheduler_args['scheduler'])
        setattr(self, 'schedulerFunc', schedulerFunc)

    def _get_input_args_from_optimizer(self):
        inputArgs = inspect.getfullargspec(self.schedulerFunc).args
        setattr(self, 'inputArgs', inputArgs)

    def _get_mutual_args(self):
        schedulerArgs = {k: self.scheduler_args[k] for k in self.scheduler_args if k in self.inputArgs}
        schedulerArgs['optimizer'] = self.optimizer
        setattr(self, 'schedulerArgs', schedulerArgs)

