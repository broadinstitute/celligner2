from torch import nn


class Agg_class(nn.Module):
    def __init__(self, enco, classi):
        super().__init__()
        self.enco = enco
        self.classi = classi

    def forward(self, dat, batch):
        means, var = self.enco(dat, batch=batch)
        return self.classi(means)
