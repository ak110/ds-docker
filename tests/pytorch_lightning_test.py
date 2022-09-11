import os

import pytest


def test_run():
    import pytorch_lightning as pl
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    assert pl.__version__ is not None

    class LitDataModule(pl.LightningDataModule):
        # pylint: disable=arguments-differ
        def __init__(self, batch_size=16):
            super().__init__()
            self.batch_size = batch_size
            self.train_ds: TensorDataset | None = None
            self.valid_ds: TensorDataset | None = None

        def setup(self, stage=None):
            X_train = torch.rand(100, 1, 28, 28)
            y_train = torch.randint(0, 10, size=(100,))
            X_valid = torch.rand(20, 1, 28, 28)
            y_valid = torch.randint(0, 10, size=(20,))
            self.train_ds = TensorDataset(X_train, y_train)
            self.valid_ds = TensorDataset(X_valid, y_valid)

        def train_dataloader(self):
            assert self.train_ds is not None
            return DataLoader(
                self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1
            )

        def val_dataloader(self):
            assert self.valid_ds is not None
            return DataLoader(
                self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=1
            )

    class LitClassifier(pl.LightningModule):
        # pylint: disable=arguments-differ
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return F.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            del batch_idx
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            del batch_idx
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log("val_loss", loss)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-2)

    dm = LitDataModule()
    model = LitClassifier()
    trainer = pl.Trainer(
        gpus=None, max_epochs=1, logger=False, enable_checkpointing=False
    )
    trainer.fit(model, datamodule=dm)
    assert "train_loss" in trainer.logged_metrics
    assert "val_loss" in trainer.logged_metrics


def test_gpu():
    """GPUのテスト。環境変数GPUに従う。"""
    import torch

    gpu = os.environ.get("GPU")
    if gpu is None:
        pytest.skip("Environment variable 'GPU' is not defined.")
    elif gpu == "none":
        assert not torch.cuda.is_available()
    else:
        assert torch.cuda.is_available()
        assert torch.cuda.get_device_name() != ""
