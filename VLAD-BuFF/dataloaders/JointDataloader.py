import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T

from dataloaders.BaiduDataset import BaiduDataset
from dataloaders.PittsburgDataset import PittsburgDataset
from . import PittsburgDataset as PittsburgDatasetModule

from prettytable import PrettyTable

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

# Supported dataset names for training
SUPPORTED_TRAIN_SETS = ["pitts", "baidu"]

# Supported validation sets
VAL_DATASETS = ["pitts30k_val", "pitts30k_test"]


class OffsetDataset(torch.utils.data.Dataset):
    """
    Wraps an existing dataset and adds an integer offset to all place_id labels.
    This prevents place_id collisions when concatenating multiple datasets
    that independently assign IDs starting from 0.
    """

    def __init__(self, dataset, offset: int):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        imgs, labels = self.dataset[index]
        return imgs, labels + self.offset

    @property
    def total_nb_images(self):
        return self.dataset.total_nb_images


class JointDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that combines multiple training datasets
    (e.g., Pitts + Baidu) into a single unified training stream, while
    supporting standard validation sets (e.g., pitts30k_val).

    Place IDs are automatically offset to prevent cross-dataset collisions
    when feeding mini-batches into the Multi-Similarity Loss / Miner.

    Uses natural (proportional) sampling via ConcatDataset — no strict
    balancing is applied, which avoids overfitting to the smaller dataset.
    """

    def __init__(
        self,
        train_set_names=None,
        batch_size=32,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        image_size=(224, 224),
        num_workers=4,
        show_data_stats=True,
        mean_std=IMAGENET_MEAN_STD,
        random_sample_from_each_place=True,
        val_set_names=None,
    ):
        super().__init__()
        if train_set_names is None:
            train_set_names = ["pitts", "baidu"]
        if val_set_names is None:
            val_set_names = ["pitts30k_val"]

        for name in train_set_names:
            if name not in SUPPORTED_TRAIN_SETS:
                raise ValueError(
                    f"Unknown train set '{name}'. Supported: {SUPPORTED_TRAIN_SETS}"
                )

        self.train_set_names = train_set_names
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.show_data_stats = show_data_stats
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        self.save_hyperparameters()

        self.train_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=mean_std["mean"], std=mean_std["std"]),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=mean_std["mean"], std=mean_std["std"]),
            ]
        )

        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": self.shuffle_all,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": False,
        }

    def _build_single_train_dataset(self, name):
        """Instantiates a training dataset by name."""
        if name == "pitts":
            return PittsburgDataset(
                split="train",
                img_per_place=self.img_per_place,
                min_img_per_place=self.min_img_per_place,
                transform=self.train_transform,
            )
        elif name == "baidu":
            return BaiduDataset(
                split="train",
                img_per_place=self.img_per_place,
                min_img_per_place=self.min_img_per_place,
                random_sample_from_each_place=self.random_sample_from_each_place,
                transform=self.train_transform,
            )

    def reload(self):
        """Builds and concatenates all training datasets with offset place_ids."""
        component_datasets = []
        current_offset = 0

        for name in self.train_set_names:
            ds = self._build_single_train_dataset(name)
            num_places = len(ds)
            wrapped = OffsetDataset(ds, offset=current_offset)
            component_datasets.append(wrapped)
            print(
                f"  [{name}] {num_places} places, "
                f"{ds.total_nb_images} images, "
                f"place_id offset = {current_offset}"
            )
            current_offset += num_places

        self.train_dataset = ConcatDataset(component_datasets)
        self._component_datasets = component_datasets

    def setup(self, stage):
        if stage == "fit":
            self.reload()

            # Load validation sets
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if valid_set_name.lower() == "pitts30k_test":
                    self.val_datasets.append(
                        PittsburgDatasetModule.get_whole_test_set(
                            input_transform=self.valid_transform
                        )
                    )
                elif valid_set_name.lower() == "pitts30k_val":
                    self.val_datasets.append(
                        PittsburgDatasetModule.get_whole_val_set(
                            input_transform=self.valid_transform
                        )
                    )
                elif valid_set_name.lower() == "msls_val":
                    print("msls_val is not supported in JointDataModule. Skipping.")
                    continue
                else:
                    print(
                        f"Validation set '{valid_set_name}' does not exist or has not been implemented yet"
                    )
                    raise NotImplementedError

            if self.show_data_stats:
                self.print_stats()

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(dataset=val_dataset, **self.valid_loader_config)
            )
        return val_dataloaders

    @property
    def total_nb_images(self):
        return sum(ds.total_nb_images for ds in self._component_datasets)

    def print_stats(self):
        print()

        table = PrettyTable()
        table.field_names = ["Dataset", "# Places", "# Images", "Place ID Offset"]
        table.align["Dataset"] = "l"
        offset = 0
        for name, ds in zip(self.train_set_names, self._component_datasets):
            n_places = len(ds)
            n_imgs = ds.total_nb_images
            table.add_row([name, n_places, n_imgs, offset])
            offset += n_places
        table.add_row(["TOTAL", len(self.train_dataset), self.total_nb_images, "-"])
        print(table.get_string(title="Joint Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i + 1}", f"{val_set_name}"])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            [
                "# of iterations per epoch",
                f"{len(self.train_dataset) // self.batch_size}",
            ]
        )
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training Config"))
