import torch


class QConfig:
    io_scale: float
    io_zp: int
    weight_scale: float
    weight_zp: int
    active: bool

    def __init__(
        self,
        io_scale: float = 1/128,
        weight_scale: float = 1/128,
        weight_zp: int = 0,
        io_zp: int = 0,
        bias_zp: int = 0,
        active=True,
    ):
        self.io_scale = io_scale
        self.io_zp = io_zp
        self.weight_scale = weight_scale
        self.weight_zp = weight_zp
        self.active = True

    def fake_quant_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.active:
            return torch.fake_quantize_per_tensor_affine(
                weights,
                self.weight_scale,
                self.weight_zp,
                -128,
                127
            )
        else:
            return weights

    def fake_quant_io(self, io: torch.Tensor) -> torch.Tensor:
        if self.active:
            return torch.fake_quantize_per_tensor_affine(
                io,
                self.io_scale,
                self.io_zp,
                -128,
                127
            )
        else:
            return io

    def fake_quant_bias(self, bias: torch.Tensor) -> torch.Tensor:
        if self.active:
            return torch.fake_quantize_per_tensor_affine(
                bias,
                self.io_scale * self.weight_scale,
                0,
                -32_768,
                32_767
            )
        else:
            return bias
