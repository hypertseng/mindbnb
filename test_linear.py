import mindspore.context
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.core import nn
from bitsandbytes.nn import Linear8bitLt


mindspore.context.set_context(device_target="GPU", pynative_synchronize=True)

p = mindspore.Parameter(Tensor(np.random.randn(512, 512).astype(np.float16)), name="weight")
int8_model = Linear8bitLt(512, 512, has_fp16_weights=False)
int8_model.to(0)

input_data = Tensor(np.random.randn(1, 512).astype(np.float16))
int8_output = int8_model(input_data)
