from sklearn.datasets import make_moons
from torch.utils.data import Dataset
import torch


class TwoMoonDataset(Dataset):

    def __init__(self, noise=0.1, n_samples=32, sample_size=256) -> None:
        super().__init__()
        self.noise = noise
        self.sample_size = sample_size
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index)    :
        X, y = make_moons(n_samples=self.sample_size, noise=self.noise)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)