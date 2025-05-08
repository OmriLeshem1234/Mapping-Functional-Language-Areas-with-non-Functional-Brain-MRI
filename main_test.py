import argparse
import torch
from pathlib2 import Path
from utils.dataloader import dataloader
from utils.model import Model
from utils.load_args import load_args
from utils.load_transforms import load_transforms
from testing.code.tester import Tester
from testing.code.post_tester_analysis import post_tester_analysis


def parse_args():
    """ Parses command-line arguments for model testing """
    parser = argparse.ArgumentParser(description="Test a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Name of model to test (swinunetr/agynet)")
    parser.add_argument("--region", type=str, required=False,
                        help="Name of region to test using agynet (wernicke/broca)")
    return parser.parse_args()


def test_experiment(conf):
    """
    Test a trained model on a test dataset based on configuration.

    Args:
        conf (str): Path to the test configuration file.
    """
    # Load arguments from configuration file
    project_dir = Path(__file__).resolve().parent
    args = load_args(conf, project_dir=project_dir)

    # Initialize device for testing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create instance of the model
    model = Model(args=args).create_model().to(device)

    # Load weights to model
    weights_path = project_dir / args.weights_path
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path), strict=True)

    # Load transforms
    load = load_transforms(args)

    # Create dataloader for testing
    Dataloader = dataloader(args=args, transform=load)
    Dataloader.create_dataloader()

    print(f"Started testing: {args.full_experiment_name}")

    # Create tester and run test loop
    tester = Tester(
        args=args,
        model=model,
        dataloader=Dataloader.test_dataloader,
        device=device
    )
    tester.test_loop()

    print(f"Finished testing: {args.full_experiment_name}\n")


if __name__ == "__main__":
    args = parse_args()

    swinunetr_configs = [f'testing/config_files/swinunetr/swinunetr_fold{i}_test.yaml' for i in range(5)]
    agynet_broca_configs = [f'testing/config_files/agynet/broca/agynet_broca_fold{i}_test.yaml' for i in range(5)]
    agynet_wernicke_configs = [f'testing/config_files/agynet/wernicke/agynet_wernicke_fold{i}_test.yaml' for i in
                               range(5)]

    if args.model == "all_models":
        configs = swinunetr_configs + agynet_broca_configs + agynet_wernicke_configs
    elif args.model == "swinunetr":
        configs = swinunetr_configs
    elif args.model == "agynet":
        if args.region == 'broca':
            configs = agynet_broca_configs
        elif args.region == 'wernicke':
            configs = agynet_broca_configs
        else:
            print(f"Error: For AGYnet, region must be either 'broca' or 'wernicke'. Received: {parse_args.region}")
            exit(1)
    else:
        print(f"Error: Model must be one of 'all_models', 'swinunetr', or 'agynet'. Received: {parse_args.model}")
        exit(1)

    for config in configs:
        test_experiment(config)

    # After all models have completed testing, perform:
    # 1. Atlas-based segmentation analysis
    # 2. ROC curve computation and AUC value calculation
    # 3. Margin sensitivity analysis
    post_tester_analysis()
