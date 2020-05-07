import random as rnd
from math import floor, ceil

from model.model import *

tf.random.set_random_seed(1950)
rnd.seed(1950)

BLOCK_LENGTH = 6
UPSCALE_BLOCKS = [1, 4]

STATE = {
    "block": ["resblock", "identity", "inv_resblock", "depth_sep_block"],
    "block_normal": [ResBlock, ConvBlock, InvertedResBlock, DepthWiseSeparateBlock],
    "block_reduction": [ResBlockUpscale, ConvBlockUpscale,
                        InvertedResBlockRUpscale, DepthWiseSeparateBlockUpscale],
    "kernel": [1, 3, 5],
    "flops": [1, 2, 4, 8]
}


def _interval_map(x, size):
    if x == 0: return 1
    return floor(ceil(x * size) / 1.0)


# noinspection PyDefaultArgument
class NeuralSearchState:
    def __init__(self, state: dict = STATE):
        self.block_state = state["block"]
        self.block_normal = state["block_normal"]
        self.block_reduction = state["block_reduction"]
        self.block_state_len = len(self.block_state)

        self.kernel_state = state["kernel"]
        self.kernel_state_len = len(self.kernel_state)

        self.flops_multiplier = state["flops"]
        self.flops_multiplier_len = len(self.flops_multiplier)

        self.size = BLOCK_LENGTH * 2 + 1

    def get_random_individual(self):
        return [rnd.random() for _ in range(BLOCK_LENGTH * 2 + 1)]

    def decode_int(self, individual: list):
        blocks, kernels = [], []
        for i, x in enumerate(individual[:BLOCK_LENGTH]):
            if i not in UPSCALE_BLOCKS:
                blocks.append(self.block_normal[x])
            else:
                blocks.append(self.block_reduction[x])

        for x in individual[BLOCK_LENGTH:BLOCK_LENGTH * 2]:
            kernels.append(self.kernel_state[x])

        x = individual[BLOCK_LENGTH * 2]
        flops = self.flops_multiplier[x]

        return blocks, kernels, flops

    def decode(self, individual: list):
        blocks, kernels = [], []
        for i, x in enumerate(individual[:BLOCK_LENGTH]):
            index = _interval_map(x, self.block_state_len)
            if i not in UPSCALE_BLOCKS:
                blocks.append(self.block_normal[index - 1])
            else:
                blocks.append(self.block_reduction[index - 1])

        for x in individual[BLOCK_LENGTH: BLOCK_LENGTH * 2]:
            index = _interval_map(x, self.kernel_state_len)
            kernels.append(self.kernel_state[index - 1])

        x = individual[BLOCK_LENGTH * 2]
        index = _interval_map(x, self.flops_multiplier_len)
        flops = self.flops_multiplier[index - 1]

        return blocks, kernels, flops

    def repr_int(self, individual: list):
        decoded = []
        for x in individual[:BLOCK_LENGTH]:
            index = _interval_map(x, self.block_state_len)
            decoded.append(index - 1)

        for x in individual[BLOCK_LENGTH:BLOCK_LENGTH * 2]:
            index = _interval_map(x, self.kernel_state_len)
            decoded.append(index - 1)

        x = individual[BLOCK_LENGTH * 2]
        index = _interval_map(x, self.flops_multiplier_len)
        decoded.append(index - 1)

        return decoded


# Test
if __name__ == '__main__':
    n = NeuralSearchState()
    ind = n.get_random_individual()
    print(ind)
    print(n.decode(ind))
    print(n.repr_int(ind))
    print(n.decode_int(n.repr_int(ind)))
