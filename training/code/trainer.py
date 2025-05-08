import numpy as np
import torch
from torch import nn
from pathlib import Path
import monai.metrics
from tqdm import tqdm
from utils.load_metrics import LoadMetrics
from utils.my_utils import *


class Trainer:
    def __init__(self,
                 args: dict,
                 model,
                 training_dataloader: monai.data.DataLoader,
                 validation_dataloader: monai.data.DataLoader,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 loss_fn: monai.losses,
                 device='cuda',
                 experiment=None):

        self.args = args
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.experiment = experiment
        self.device = device
        self.epoch = 0  # epoch counter
        self.weights_path = getattr(self.args, 'save_weights_path', 'None')
        self.checkpoints_path = getattr(self.args, 'save_checkpoints_path', 'None')
        self.final_activation = getattr(self.args, 'final_activation', False)  # Optional final activation for model
        self.amp_dtype_str = getattr(self.args, 'amp_dtype', None)
        self.amp_dtype = getattr(torch, self.amp_dtype_str, None)
        self.metrics_obj = None
        self.metric_value_max = -np.inf
        self.use_pretrained_hcp = getattr(self.args, 'use_pretrained_hcp', False)
        self.pretrained_hcp_path = getattr(self.args, 'pretrained_hcp_path', None)
        if self.use_pretrained_hcp:
            state_dict = torch.load(Path(self.args.project_dir) / self.pretrained_hcp_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)

    def model_improves(self):
        # Metric computed to understand if model improved
        mean_value = np.mean(np.concatenate(self.metrics_values['validation'][self.args.metricForSaving]).ravel())
        if mean_value > self.metric_value_max:
            self.metric_value_max = mean_value
            return True
        else:
            return False

    def log(self, name, value, key='training', step=0):
        if self.experiment is not None:
            self.experiment.log_metric(name=name + f'_{key}',
                                       value=value,
                                       step=step)

    def log_metrics(self):
        for key in self.metrics_values:
            for metric in self.metrics_values[key]:
                try:
                    mean_value = np.mean(np.concatenate(self.metrics_values[key][metric]).ravel())
                except ValueError:
                    mean_value = np.mean(np.array(self.metrics_values[key][metric]))
                finally:
                    self.log(name=metric, value=mean_value, key=key, step=self.epoch)

    def naught_metrics_values(self):
        self.metrics_values = {'training': {k: [] for k in self.metrics_obj.metrics},
                               'validation': {k: [] for k in self.metrics_obj.metrics}}

    def load_metrics(self):
        self.metrics_obj = LoadMetrics(self.args)
        self.metrics_obj.create_metrics()

        # Creating a dictionary to keep metrics values
        self.metrics_values = {'training': {k: [] for k in self.metrics_obj.metrics},
                               'validation': {k: [] for k in self.metrics_obj.metrics}}

        # Create variable to keep score of optimized MetricForSaving
        self.max_metric = getattr(self.args, 'metric_value_max', None)

    def compute_metrics(self, y_pred, y, key='training'):
        y, y_pred = y.clone().detach().cpu().to(torch.float64), y_pred.clone().detach().cpu().to(torch.float64)
        for metricName, metricFunc in self.metrics_obj.metrics.items():
            metric_value = metricFunc(y_pred=y_pred, y=y).numpy().astype(np.float64)
            if np.isnan(metric_value).any():
                print(f" metric {metricName}, key {key} of some batch is NaN")
            self.metrics_values[key][metricName].append(metric_value)

    def save_weights(self):
        path = Path(
            f'{self.args.project_dir}/{self.weights_path}/{self.args.experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        torch.save(obj=self.model.state_dict(),
                   f=path / f'best_model.pt'
                   )

    def save_checkpoint(self, best=False, last=False):
        path = Path(
            f'{self.args.project_dir}/{self.checkpoints_path}/{self.args.experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        obj = {
            'config': self.args,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.validation_loss,
            'metric_name': self.args.metricForSaving,
            'metric_value_max': self.metric_value_max,
        }
        obj = obj.update({'scheduler_state_dict': self.scheduler.state_dict()}) if self.scheduler is not None else obj
        if last:
            f = f'{path}/last_checkpoint.pt'
            torch.save(obj=obj, f=f)
        else:
            f = f'{path}/best_checkpoint.pt' if best else f'{path}/epoch{self.epoch}.pt'
            torch.save(obj=obj, f=f)

    @staticmethod
    def probability2onehot(y_pred):
        """
        Convert the model's output to one-hot encoding.
        """

        y_pred_seg = torch.permute(
            torch.nn.functional.one_hot(
                torch.argmax(y_pred, dim=1),
                num_classes=y_pred.shape[1]
            ),
            dims=(0, 4, 1, 2, 3)
        ).to(torch.int)

        return y_pred_seg

    def eval_step(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)

        with torch.autocast('cuda', dtype=self.amp_dtype):
            with torch.no_grad():
                batch["y_pred"] = self.model(batch["img"])

                # Make sure there are no NaNs
                if torch.isnan(batch["y_pred"]).any():
                    print('Eval Step: There are NaNs')
                    return -1

                if self.final_activation:
                    batch["y_pred"] = torch.nn.functional.softmax(batch["y_pred"], dim=1)

                # Convert the model's output to one-hot encoding
                batch["y_pred_seg"] = self.probability2onehot(batch["y_pred"])

                # Compute the validation loss
                loss = self.loss_fn(pred=batch["y_pred"], gt=batch["gt"], gm=batch["gm"], model=self.model)

        # Compute metrics
        for i in range(batch["gt"].shape[0]):
            self.compute_metrics(y=torch.unsqueeze(batch["gt"][i], 0), y_pred=batch["y_pred_seg"][i].unsqueeze(0),
                                 key='validation')

        return loss.item()

    def eval(self):
        # Change model to eval mode
        self.model.eval()
        running_loss_val = []

        for batch_idx, batch in enumerate(self.validation_dataloader):
            # Validation step
            loss_val = self.eval_step(batch)
            running_loss_val.append(loss_val)

        # self.model.train()
        self.model.train()
        return np.mean(running_loss_val)

    def train_step(self, batch):

        for key in batch:
            batch[key] = batch[key].to(self.device)

        # Calculating segmentation prediction from model & one-hot vectoring
        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
            batch["y_pred"] = self.model(batch["img"])

            # Make sure there are no NaNs
            if torch.isnan(batch["y_pred"]).any():
                print('train Step: There are NaNs')
                return -1

            if self.final_activation:
                batch["y_pred"] = torch.nn.functional.softmax(batch["y_pred"], dim=1)

            # Convert the model's output to one-hot encoding
            batch["y_pred_seg"] = self.probability2onehot(batch["y_pred"])

            # Compute the loss and its gradients
            loss = self.loss_fn(pred=batch["y_pred"], gt=batch["gt"], gm=batch["gm"], model=self.model)

        # Scaler backward
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        torch.cuda.empty_cache()

        # Compute metrics
        for i in range(batch["gt"].shape[0]):
            self.compute_metrics(y=torch.unsqueeze(batch["gt"][i], 0), y_pred=batch["y_pred_seg"][i].unsqueeze(0),
                                 key='training')

        return loss.item()

    def train_loop(self):

        # Load metrics
        self.load_metrics()

        # Assign model to training mode
        self.model.train()

        # GradScaler
        self.scaler = torch.amp.GradScaler('cuda')

        for epoch in range(1, self.args.epochs + 1):
            self.epoch += 1  # Keep epoch number
            loader = tqdm(self.training_dataloader)  # Wrap loader with tqdm
            self.naught_metrics_values()  # Restart metrics values
            running_loss = []

            for batch_idx, batch in enumerate(loader):
                self.optimizer.zero_grad(set_to_none=True)

                # train loop & compute metrics on training
                loss = self.train_step(batch)

                # loss tracking
                loader.set_description(f'Epoch {self.epoch}: batch loss = {loss:.4f}')
                running_loss.append(loss)

            # Training loss
            loss_mean = np.mean(running_loss)
            self.training_loss = loss_mean
            self.log('loss', self.training_loss, key='training', step=self.epoch)

            # Checking on Validation set
            loss_val = self.eval()
            self.validation_loss = loss_val
            self.log('loss', self.validation_loss, key='validation', step=self.epoch)

            # Update and log scheduler
            if self.scheduler is not None:
                self.log(name='Learning Rate', value=self.scheduler.get_last_lr(), key='', step=self.epoch)
                self.scheduler.step()

            # Log metrics
            self.log_metrics()

            # Save checkpoints
            self.save_checkpoint(best=False, last=True)

            # Save state_dict if model improves
            if self.model_improves():
                self.save_weights()
                self.save_checkpoint(best=True, last=False)

        if self.experiment is not None:
            self.experiment.end()
