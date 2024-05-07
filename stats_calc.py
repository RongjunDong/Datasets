import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def options() -> argparse.Namespace:
    """
    Parse the command line options

    Returns
    -------
    argparse.Namespace
        The command line options
    """
    parser = argparse.ArgumentParser(description='Calculate statistics for a given dataset')
    parser.add_argument(
            "--images_folder",
            type=str,
            required=False,
            default="./Topographies/raw/FiguresStacked 8X8",
            help="path to the images",
        )
    parser.add_argument(
            "--label_path",
            type=str,
            required=False,
            default="./biology_data/TopoChip/AeruginosaWithClass.csv",
            help="path to the label csv file",
        )
    parser.add_argument(
            "--image_type",
            type=str,
            required=False,
            default="png",
            help="type of the images",
        )
    parser.add_argument(
        "--image_size",
        type=int,
        required=False,
        default=224,
        help="size of the images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=2200,
        help="batch size",
    )

    args = parser.parse_args()
    for k, v in vars(args).items():
            print(f"{str(k)}: {str(v)}")

    return args

def calculate_mean_std(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and standard deviation of the dataset

    Parameters
    ----------
    loader: DataLoader
        The data loader

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The mean and standard deviation of the dataset per channel
    """
    mean = 0.0
    std = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for images, _ in loader:
        images = images.to(device)
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std

def save_results(mean: float, std: float, args: argparse.Namespace) -> None:
    print(f"Mean: {mean}, Std: {std}")
    results = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "args": vars(args)
    }
    results_path = Path("stats_results")
    results_path.mkdir(parents=True, exist_ok=True)
    file_name = f"stats_{args.images_folder.split('/')[-1]}.json"
    with open(results_path / file_name, "w") as file:
        json.dump(results, file, indent=4)
        
    print(f"Results saved to {results_path / file_name}")


class TopographiesDataset(Dataset):
    """
    Biological Observation dataset class
    """

    def __init__(self, args) -> None:
        """
        Initializes the BiologicalObservation class
        """
        super().__init__()
        self._images_folder = args.images_folder
        self._image_type = args.image_type
        self._image_size = args.image_size
        self._label_path = args.label_path

        self._annotations = pd.read_csv(self._label_path)
        self._images_names = [
            Path(image).stem
            for image in glob.glob(
                os.path.join(self._images_folder, f"*.{self._image_type}")
            )
        ]
        self._featids = [image.split("_")[2] for image in self._images_names]
        self.transform = transforms.Compose([
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor()
        ])
        
        
    def __getitem__(self, index):
        """
        Returns the data at the given index

        Parameters
        ----------
        index: int
            The index of the data to return

        Returns
        -------
        image: torch.Tensor
            The image at the given index
        """
        featid, label = self._annotations.iloc[index]
        image_name = self._images_names[self._featids.index(str(featid))]
        img_path = os.path.join(self._images_folder, f"{image_name}.{self._image_type}")

        image = Image.open(img_path)
        image = self.transform(image)
        label = torch.tensor(label)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self._annotations)
    



if __name__ == '__main__':
    args = options()
    dataset = TopographiesDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    mean, std = calculate_mean_std(dataloader)
    save_results(mean, std, args)