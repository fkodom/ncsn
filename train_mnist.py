"""
train_mnist.py
--------------
"""

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ncsn import MnistTrainingModule
from ncsn.utils.logger import custom_mlflow_logger
from ncsn.utils.visualize import *


def main(
    experiment: str = "mnist",
    run: str = None,
    gpus: int = 0,
    optimizer: str = "Adam",
    lr: float = 1e-3,
    epochs: int = 25,
    batch_size: int = 150,
    checkpoint: str = None,
    eval_: bool = False,
    debug: bool = True,
):
    ncsn = MnistTrainingModule(
        experiment=experiment,
        run=run,
        optimizer=optimizer,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    if args.checkpoint is not None:
        ncsn.load_from_checkpoint(args.checkpoint)

    if eval_:
        ncsn.eval()
        if gpus:
            ncsn.cuda()
        visualize_random_samples(ncsn)
    else:
        if not debug:
            checkpoint_callback = ModelCheckpoint(
                prefix=f"{experiment}_{run}_", filepath="models", monitor="loss",
            )
        else:
            checkpoint_callback = None

        pl.Trainer(
            gpus=gpus,
            max_epochs=epochs,
            resume_from_checkpoint=checkpoint,
            logger=custom_mlflow_logger(
                experiment_name=experiment,
                run_name=run,
                debug=debug,
            ),
            checkpoint_callback=checkpoint_callback,
            weights_summary=None,
        ).fit(ncsn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--experiment", default="mnist")
    parser.add_argument("-r", "--run", default=None)
    parser.add_argument("-g", "--gpus", default=0)
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-l", "--lr", default=1e-3)
    parser.add_argument("-e", "--epochs", default=25)
    parser.add_argument("-c", "--checkpoint", default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(
        experiment=args.experiment,
        run=args.run,
        gpus=int(args.gpus),
        optimizer=args.optimizer,
        lr=float(args.lr),
        epochs=int(args.epochs),
        checkpoint=args.checkpoint,
        eval_=args.eval,
        debug=args.debug,
    )
