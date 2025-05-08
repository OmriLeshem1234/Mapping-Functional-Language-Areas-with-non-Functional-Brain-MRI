import argparse
import torch
from pathlib2 import Path
from comet_ml import Experiment
from utils.dataloader import dataloader
from utils.load_args import load_args
from utils.loss_functions import TrainingLoss
from utils.load_transforms import load_transforms
from utils.model import Model
from utils.optimizer import optimizer as Optimizer
from utils.Scheduler import Scheduler
from training.code.trainer import Trainer


def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--model", type=str, required=True, help="Name of model to train (swinunetr/agynet)")
    parser.add_argument("--region", type=str, required=False, help="Name of region to train using agynet")
    return parser.parse_args()


def train_experiment(conf):
    # Load arguments from configuration file
    project_dir = Path(__file__).resolve().parent
    args = load_args(conf, project_dir=project_dir)

    # Initialize experiment name
    Name = f'{args.project_name}{args.experiment_number}_{args.experiment_name}'
    if args.log:
        experiment = Experiment(
            api_key=args.api_key,
            project_name=args.project_name,
            workspace=args.workspace,
        )
        experiment.set_name(name=Name)
    else:
        experiment = None

    # Initialize device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load all transforms to one object
    load = load_transforms(args)

    # Creating dataloader
    Dataloader = dataloader(args=args, transform=load)
    Dataloader.create_dataloader()

    # Creating instance of the model
    model = Model(args=args).create_model().to(device)

    # Assigning optimizer and its inputs
    optimizer = Optimizer(args, model=model).create_optimizer()

    # Creating a scheduler
    scheduler = Scheduler(args, optimizer).create_scheduler()

    # Assigning loss function
    _lambda = 1 if args.anatomy_loss else 0
    lossFunc = TrainingLoss(_lambda=_lambda, weight_decay_factor=200)

    print(f"Started training: {args.full_experiment_name}")

    train = Trainer(args=args,
                    model=model,
                    training_dataloader=Dataloader.training_dataloader,
                    validation_dataloader=Dataloader.validation_dataloader,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=lossFunc,
                    experiment=experiment)
    train.train_loop()

    print(f"Finished training: {args.full_experiment_name}")


if __name__ == "__main__":
    args = parse_args()

    swinunetr_configs = [f'training/config_files/swinunetr/swinunetr_fold{i}.yaml' for i in range(5)]
    agynet_broca_configs = [f'training/config_files/agynet/broca/agynet_broca_fold{i}.yaml' for i in range(5)]
    agynet_wernicke_configs = [f'training/config_files/agynet/wernicke/agynet_wernicke_fold{i}.yaml' for i in range(5)]

    if args.model == "all_models":
        configs = swinunetr_configs + agynet_broca_configs + agynet_wernicke_configs
    elif args.model == "swinunetr":
        configs = swinunetr_configs
    elif args.model == "agynet":
        if args.region == 'broca':
            configs = agynet_broca_configs
        elif args.region == 'wernicke':
            configs = agynet_wernicke_configs
        else:
            print(f"Error: For AGYnet, region must be either 'broca' or 'wernicke'. Received: {args.region}")
            exit(1)
    else:
        print(f"Error: Model must be one of 'all_models', 'swinunetr', or 'agynet'. Received: {args.model}")
        exit(1)

    for config in configs:
        train_experiment(config)
