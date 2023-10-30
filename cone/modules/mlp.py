import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hid: int,
        num_layers: int,
        dropout: float = 0.0,
        layernorm: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_dim_in = dim_in if i == 0 else dim_hid
            layer_dim_out = dim_out if i == num_layers - 1 else dim_hid

            layer = nn.Sequential(
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU(),
                nn.LayerNorm(layer_dim_out) if layernorm else nn.Identity(),
                nn.Dropout(dropout),
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
