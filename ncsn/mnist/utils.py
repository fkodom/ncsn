from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor

from ncsn.utils.data import format_dataset


def load_mnist_data() -> Dataset:
    return format_dataset(
        MNIST(
            "data",
            transform=Compose([Resize((16, 16)), ToTensor()]),
            download=True
        ),
        start_sigma=1.0,
        end_sigma=0.01
    )
