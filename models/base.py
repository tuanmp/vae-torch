from pytorch_lightning import LightningModule
import torch


class BaseVAE(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def encode(self, x, cond=None):
        raise NotImplementedError

    def decode(self, z, cond=None):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams['learning_rate'])
