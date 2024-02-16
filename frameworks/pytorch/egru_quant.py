import numpy as np


def quantize(val, bits, factor=1):
    scale = 1 << (bits - 1)
    return (
        np.round((val * scale) / factor)
        .clip(
            -scale,
            scale - 1,
        )
        .astype(np.int32)
    )


def dequantize(val, bits):
    return val.astype(np.float64) / (1 << (bits - 1))


def requantize(val, shift, bits, factor=1):
    assert val.dtype == np.int32
    return ((val * factor) >> shift).clip(
        -(1 << (bits - 1)),
        (1 << (bits - 1)) - 1,
    )


class EGRUQuant:
    kernel: np.ndarray
    bias: np.ndarray
    rec_kernel: np.ndarray
    rec_bias: np.ndarray
    thr: np.ndarray
    state: np.ndarray
    weight_scale: int
    prev_output: np.ndarray
    weight_bits: int
    activation_bits: int
    resest_gate_scale: int

    def __init__(
        self,
        kernel: np.ndarray,
        bias: np.ndarray,
        rec_kernel: np.ndarray,
        rec_bias: np.ndarray,
        thr: np.ndarray,
        weight_scale: int = 1,
        weight_bits: int = 6,
        activation_bits: int = 6,
        resest_gate_scale: int = 4,
    ):
        self.weight_scale = weight_scale
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.resest_gate_scale = resest_gate_scale
        self.kernel = quantize(kernel, weight_bits, weight_scale)
        self.bias = quantize(bias, weight_bits + activation_bits, weight_scale * 2)
        self.rec_kernel = quantize(rec_kernel, weight_bits, weight_scale)
        self.rec_bias = quantize(
            rec_bias, weight_bits + activation_bits, weight_scale * 2
        )
        self.thr = quantize(thr, activation_bits)
        self.state = np.zeros_like(self.thr)

    def reset(self, prev_output: np.ndarray):
        self.state = np.zeros_like(self.thr)
        self.prev_output = quantize(prev_output, self.activation_bits)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        assert input.dtype == np.int32
        # same as self.kernel.transpose() @ input
        vx = np.split(input @ self.kernel + self.bias, 3, 1)
        vh = np.split(self.prev_output @ self.rec_kernel + self.rec_bias, 3, 1)

        z = self.sigmoid(vx[0] + vh[0])
        r = self.sigmoid(vx[1] + vh[1])
        g_prime = requantize(
            vh[2],
            self.weight_bits + 1,
            self.activation_bits,
            self.weight_scale,
        )
        g = self.tanh(vx[2] + ((r * g_prime) << 2))
        cur_h = requantize(
            z * self.state + (quantize(1, self.activation_bits) - z) * g,
            self.activation_bits - 1,
            self.activation_bits,
        )
        sparsity = np.sum(cur_h >= self.thr) / len(self.thr)
        output = np.where(cur_h >= self.thr, cur_h, 0)
        self.state = cur_h - np.where(cur_h >= self.thr, self.thr, 0)
        self.prev_output = output
        return output

    def sigmoid(self, val: np.ndarray) -> np.ndarray:
        off = 1 << (self.activation_bits - 2)
        val = requantize(
            val, self.weight_bits + 1, self.activation_bits - 1, self.weight_scale
        )
        return val + off

    def tanh(self, val: np.ndarray) -> np.ndarray:
        return requantize(
            val,
            self.weight_bits - 1,
            self.activation_bits,
            self.weight_scale,
        )
