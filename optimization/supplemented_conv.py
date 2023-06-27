import torch
from torch.nn import Conv2d
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d

class SupplementedConv2d(Conv2d):

    def __init__(
        self,
        base_conv: Conv2d,
    ) -> None:

        super().__init__(
            base_conv.in_channels,
            base_conv.out_channels,
            base_conv.kernel_size,
            base_conv.stride,
            base_conv.padding,
            base_conv.dilation,
            base_conv.groups,
            (base_conv.bias is not None),
            base_conv.padding_mode,
        )

        self.register_buffer('w1', base_conv.weight.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.weight + self.w1, self.bias)


class SupplementedDecomposedConv2d(DecomposedConv2d):
    def __init__(
        self,
        base_conv: DecomposedConv2d,
        decomposing_mode: str = 'channel',
        n: int = 1
    ) -> None:

        super().__init__(
            base_conv=base_conv,
            decomposing_mode=decomposing_mode
        )

        S, indices = base_conv.S.sort()
        U = base_conv.U[:, indices]
        Vh = base_conv.Vh[indices, :]

        base_conv.compose()
        self.register_buffer('w1', base_conv.weight.data)
        with torch.no_grad():
            self.S[:n] = S[:n]
            self.U[:, :n] = U[:, :n]
            self.Vh[:n, :] = Vh[:n, :]


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if self.decomposing:
            W = self.U @ torch.diag(self.S) @ self.Vh
            return self._conv_forward(
                input,
                W.view(
                    self.out_channels,
                    self.in_channels // self.groups,
                    *self.kernel_size
                ) + self.w1,
                self.bias,
            )
        else:
            return self._conv_forward(input, self.weight+self.w1, self.bias)
