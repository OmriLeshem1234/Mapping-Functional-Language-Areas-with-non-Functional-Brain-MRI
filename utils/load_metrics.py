"""
This file is a metrics class which loads metrics from MONAI and self-made metrics.
"""

import monai
from utils import my_metrics
from inspect import getfullargspec, isclass


class LoadMetrics:
    def __init__(self, args):
        self.args = args
        self.metrics_args = self.args['metrics']
        self.metrics_names = self.metrics_args.keys()
        self.metrics = {}

    def _import_metric_from_monai(self, metric_name):
        metricFunc = getattr(monai.metrics, metric_name)
        inputArgs = getfullargspec(metricFunc).args
        metricArgs = {k: self.metrics_args[f'{metric_name}'][k] for k in self.metrics_args[f'{metric_name}'] if k in inputArgs}
        return metricFunc(**metricArgs)

    def _import_metric_from_my_metrics(self, metric_name):
        metricFunc = getattr(my_metrics, metric_name)
        inputArgs = getfullargspec(metricFunc).args
        metricArgs = {k: self.metrics_args[f'{metric_name}'][k] for k in self.metrics_args[f'{metric_name}'] if k in inputArgs}
        return metricFunc(**metricArgs)

    def create_metrics(self):
        monai_metric_classes = [attr for attr in dir(monai.metrics) if isclass(getattr(monai.metrics, attr))]
        for metric in self.metrics_names:
            if metric in monai_metric_classes:
                self.metrics[metric] = self._import_metric_from_monai(metric_name=metric)
            elif metric in self.args.metrics:
                self.metrics[metric] = self._import_metric_from_my_metrics(metric_name=metric)