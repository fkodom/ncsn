"""
train_celeba.py
--------
"""

import os

from tiki import Trainer
from ncsn.celeba import CelebaConv, load_celeba_data, load_celeba_small_data
from ncsn.utils.visualize import *


if __name__ == "__main__":
    # -------------------------- Runtime Parameters --------------------------
    # model: str = ""
    model: str = os.path.join("models", "CelebaConv.dict")
    train: bool = False
    epochs: int = 50
    batch_size: int = 250
    # ------------------------------------------------------------------------

    ncsn = CelebaConv()
    if model:
        ncsn.load_state_dict(torch.load(model, map_location="cpu"))

    if train:
        print("Loading data... ", end="")
        tr_dataset = load_celeba_data()
        print("done!")

        Trainer().train(
            ncsn,
            tr_dataset=tr_dataset,
            loss="smooth_l1",
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
        device = "cuda:0"
        ncsn.to(device)

    while True:
        # idx = torch.randperm(len(tr_dataset))[:9]
        # labels = torch.stack([tr_dataset[i.item()][1] for i in idx], 0)
        # labels = labels.to(device)
        # samples = ncsn.sample_from(labels, return_all=True, identical_noise=True)
        # samples = (samples - samples.min()) / (samples.max() - samples.min())
        # visualize_samples(samples[:, :-25], size=(3, 3))
        visualize_random_samples(ncsn, num_classes=40, size=(4, 4))
