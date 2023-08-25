import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from src.data.BatchSize import BatchSize


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: BatchSize,
        num_workers: int = 0,
        data_dir: str = "./data",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size.train,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size.val,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size.test,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size.test,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False,
        )
