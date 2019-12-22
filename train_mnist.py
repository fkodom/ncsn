"""
train_mnist.py
--------
"""

import os

from torch.utils.data import ConcatDataset

from tiki import Trainer
from ncsn.mnist import MnistConv, load_mnist_data
from ncsn.utils.visualize import *


if __name__ == "__main__":
    # -------------------------- Runtime Parameters --------------------------
    # model: str = ""
    model: str = os.path.join("models", "MnistConv.dict")
    train: bool = False
    epochs: int = 25
    batch_size: int = 100
    # ------------------------------------------------------------------------

    ncsn = MnistConv()
    if model:
        ncsn.load_state_dict(torch.load(model, map_location="cpu"))

    if train:
        print("Loading data... ", end="")
        tr_dataset = ConcatDataset([load_mnist_data() for _ in range(5)])
        print("done!")

        Trainer().train(
            ncsn,
            tr_dataset=tr_dataset,
            loss="mse",
            optimizer="adam",
            gpus=[0],
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=[
                "terminate_on_nan",
                "model_checkpoint",
                "tiki_hut"
            ],
        )

    ncsn.eval()
    if torch.cuda.is_available():
        ncsn.cuda()

    while True:
        visualize_random_samples(ncsn)
        # visualize_class_samples(ncsn, 3)
        # visualize_class_iterations(ncsn, 0, 9)
