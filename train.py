import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

from src.data.BatchSize import BatchSize
from src.data.MNIST import MNISTDataModule
from src.model.AutoEncoder import LitAutoEncoder


def main(
    project_name: str,
    lr: float,
    epochs: int,
    accumulate_grad_batches: int,
    num_workers: int,
    is_barebone: bool = False,
):
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[2, 5, 6, 7],
        strategy="ddp",
        precision="16-mixed",
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=get_logger(
            project_name=project_name,
            is_barebone=is_barebone,
        ),
        log_every_n_steps=50 if not is_barebone else None,
        gradient_clip_val=0.5,
        plugins=[],
        callbacks=[
            EarlyStopping("val_loss"),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
        ]
        if not is_barebone
        else [],
        barebones=is_barebone,
        deterministic=False,  # True: ensures reproducibility
        fast_dev_run=False,
        profiler=None,
    )

    trainer.fit(
        model=LitAutoEncoder(
            lr=lr,
        ),
        datamodule=MNISTDataModule(
            batch_size=BatchSize(
                train=128,
                val=256,
                test=256,
            ),
            num_workers=num_workers,
        ),
    )


def get_logger(
    project_name: str,
    is_barebone: bool,
):
    return (
        WandbLogger(
            project=project_name,
            save_dir=".log",
        )
        if not is_barebone
        else None
    )


if __name__ == "__main__":
    L.seed_everything(42, workers=True)
    main(
        project_name="pixi-lightning",
        lr=1e-4,
        epochs=2,
        accumulate_grad_batches=1,
        num_workers=4,
        is_barebone=False,
    )
