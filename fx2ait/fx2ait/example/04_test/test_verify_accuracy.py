import unittest

import torch
import torch.nn as nn

from fx2ait.example.benchmark_utils import benchmark_function, verify_accuracy

in_sz = 8*8

class fcn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.ln0 = nn.Linear(in_sz,32)
        self.ln4 = nn.Linear(32,10)
    def forward(self, x: torch.Tensor):
        out0 = self.relu(self.ln0(x))
        out0 = self.softmax(self.ln4(out0))
        return out0

class TestResNet(unittest.TestCase):

    def test_model(self):
        torch.set_grad_enabled(False)
        torch.manual_seed(0)

        model = fcn().cuda().half().eval()

        inputs = [torch.randn(1, in_sz).half().cuda()]
        y = model(inputs[0])
        print("== Test result:", y)

        print("== Running verify_accuracy")
        verify_accuracy(
            model,
            inputs,
        )
        print("== Verify_accuracy done")

        # results = []
        # for batch_size in [1, 8, 16, 32, 256, 512]:
        #     inputs = [torch.randn(batch_size, 3, 224, 224).half().cuda()]
        #     results.append(
        #         benchmark_function(
        #             self.__class__.__name__,
        #             100,
        #             model,
        #             inputs,
        #         )
        #     )
        # for res in results:
        #     print(res)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
