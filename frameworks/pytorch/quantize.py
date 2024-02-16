import torch
import numpy as np


def quantize(val, bits, factor=1):
    scale = 1 << (bits - 1)
    res = (
        np.round((val * scale) / factor)
        .clip(
            -scale,
            scale - 1,
        )
        .astype(np.int32)
    )
    return res


def dequantize(val, bits):
    return val.type(torch.float) / (1 << (bits - 1))


class QConfig:
    activation_bits: int = 6,
    weights_bits: int = 6,
    weight_scale: int = 1,
    reset_gate_scale: int = 4,
    active: bool
    fake_quant: bool
    # max_val = 0

    def __init__(
        self,
        activation_bits: int = 6,
        weights_bits: int = 6,
        weight_scale: int = 1,
        reset_gate_scale: int = 4,
        active=True,
    ):
        self.activation_bits = activation_bits
        self.weights_bits = weights_bits
        self.weight_scale = weight_scale
        self.reset_gate_scale = reset_gate_scale
        self.active = True
        self.fake_quant = False

    def fake_quant_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.active:
            scale = 1 << (self.activation_bits - 1)
            res = torch.fake_quantize_per_tensor_affine(
                weights,
                self.weight_scale/scale,
                0,
                -scale,
                scale - 1,
            )
            # assert torch.min(torch.abs(res[res != 0])) >= 1/self.weight_scale
            # assert torch.max(torch.abs(res)) <= 1
            return res
        else:
            return weights

    def dequant(self, io: torch.Tensor) -> torch.Tensor:
        if not self.active or self.fake_quant:
            return io
        return dequantize(io, self.activation_bits)

    def fake_quant_io(self, io: torch.Tensor) -> torch.Tensor:
        if self.active:
            if self.fake_quant:
                scale = 1 << (self.activation_bits - 1)
                res = torch.fake_quantize_per_tensor_affine(
                    io, 1 / scale, 0, -scale, scale - 1
                )
            else:
                res = quantize(io.cpu().detach().numpy(), self.activation_bits)
                res = torch.from_numpy(res)
            # assert torch.min(torch.abs(res[res != 0])) >= 1/self.io_scale
            # assert torch.max(torch.abs(res)) <= 1
            return res
        else:
            return io

    def fake_quant_rec_candidate_activaton(self, io: torch.Tensor) -> torch.Tensor:
        if self.active:
            # max_val = torch.max(torch.abs(io))
            # if max_val > QConfig.max_val:
            #     print(max_val)
            #     QConfig.max_val = max_val
            scale = 1 << (self.activation_bits - 1)
            res = torch.fake_quantize_per_tensor_affine(
                    io, self.reset_gate_scale / scale, 0, -scale, scale - 1
            )
            # assert torch.min(torch.abs(res[res != 0])) >= 1/self.io_scale
            # assert torch.max(torch.abs(res)) <= 1
            return res
        else:
            return io

    def fake_quant_bias(self, bias: torch.Tensor) -> torch.Tensor:
        if self.active:
            scale = 1 << (self.activation_bits - 1 + self.weights_bits - 1)
            return torch.fake_quantize_per_tensor_affine(
                    bias, self.reset_gate_scale / scale, 0, -32_768, 32_767
            )
        else:
            return bias
