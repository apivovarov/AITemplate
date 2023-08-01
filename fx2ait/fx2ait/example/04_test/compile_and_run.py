import torch
import torch.nn as nn
print("cuda", torch.cuda.is_available())

import aitemplate
print("ait good")

from fx2ait.acc_tracer import acc_tracer
from fx2ait.ait_module import AITModule
from fx2ait.fx2ait import AITInterpreter
print("fx2ait good")

import uuid

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

model = fcn().cuda().half().eval()

inputs = [torch.randn(1, in_sz).half().cuda()]

mod = acc_tracer.trace(
    model,
    inputs,
)

interp = AITInterpreter(
        mod,
        inputs,
        "/tmp",
        f"test-fx2ait-{uuid.uuid1()}",
)

with torch.no_grad():
    print("== Running AITInterpreter...")
    interp_result = interp.run()
    print(f"== Running AITInterpreter done. {interp_result.engine.lib_path=}")
    ait_mod = AITModule(
        torch.classes.ait.AITModel(
            interp_result.engine.lib_path,
            interp_result.input_names,
            interp_result.output_names,
            torch.float16,
            torch.float,
            1,  #  num_runtimes
        )
    )

    print("== Running AIT model:")
    outputs = ait_mod(*inputs)
    if isinstance(outputs, torch.Tensor):
      outputs = outputs.detach().cpu()
    print("== Outputs:")
    print(outputs)
