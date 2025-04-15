from .base import BaseVAE
from torch import nn
import torch


class VAE(BaseVAE):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.x_dim, self.sample_dim, self.cond_dim = hparams['x_dim'], hparams['sample_dim'], hparams['cond_dim']
        encoder = []
        decoder = []
        for in_dim, out_dim in zip([self.x_dim + self.cond_dim] + hparams['enc_hidden_dim'][:-1], hparams['enc_hidden_dim']) :
            encoder.append(nn.Linear(in_dim, out_dim))
            encoder.append(getattr(nn, hparams['hidden_activation'])())
        encoder.append(nn.Linear(hparams['enc_hidden_dim'][-1], self.sample_dim * 2))

        for in_dim, out_dim in zip([self.sample_dim + self.cond_dim] + hparams['dec_hidden_dim'][:-1], hparams['dec_hidden_dim']) :
            decoder.append(nn.Linear(in_dim, out_dim))
            decoder.append(getattr(nn, hparams['hidden_activation'])())
        decoder.append(nn.Linear(hparams['dec_hidden_dim'][-1], self.x_dim))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        self.batch_size = hparams['batch_size']
        self.n_samples = hparams.get("n_samples") or 1
    
    def encode(self, x, cond=None):
        if cond is not None:
            x_in = torch.concat([x, cond], dim=-1)
        
        return self.encoder(x_in)
    
    def decode(self, sample, cond=None):
        if cond is not None:
            x_in = torch.concat([sample, cond], dim=-1)
        
        return self.decoder(x_in)

    def sample_prior(self, cond):
        z = torch.rand((cond.size(0), self.sample_dim)).to(self.device)

        return self.decode(z, cond)
    
    def forward(self, x, cond=None):

        encoded_ = self.encode(x, cond)

        mu_z = encoded_[:, : self.sample_dim]

        log_var_z = encoded_[:, self.sample_dim :]

        sample = self.reparameterize(mu_z, log_var_z)

        decoded_ = self.decode(sample, cond)

        return decoded_, mu_z, log_var_z

    def loss_function(self, X_hat, mu_z, log_var_z, X):

        kl_div = 0.5 * torch.mean(torch.sum(log_var_z - torch.exp(log_var_z) - torch.pow(mu_z, 2), dim=-1), dim=0) # ommit the constant 1

        reco_loss = 0.5 * torch.sum((X-X_hat)**2 )

        loss = - kl_div + reco_loss 

        return kl_div, reco_loss, loss

    def training_step(self, batch, batch_idx):

        X, y = batch[0]
        
        X, y = X[0], y[0].view(-1,1)
        
        x_out, mu_z, log_var_z = self(X, y)

        kl_div, reco_loss, loss = self.loss_function(x_out, mu_z, log_var_z, X)

        self.log_dict({
            "train_loss": loss,
            "train_kl_div": kl_div,
            "train_reco_loss": reco_loss
        }, on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
    
        return loss

    def validation_step(self, batch, batch_idx):
    
        X, y = batch

        X, y = X[0], y[0].view(-1,1)

        x_out, mu_z, log_var_z = self(X, y)

        kl_div, reco_loss, loss = self.loss_function(x_out, mu_z, log_var_z, X)

        self.log_dict({
            "val_loss": loss,
            "val_kl_div": kl_div,
            "val_reco_loss": reco_loss
        }, on_step=False, on_epoch=True, batch_size=self.batch_size)
    
        return loss
    