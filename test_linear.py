import mindspore.context
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.core import nn
from bitsandbytes.nn import Linear8bitLt


np.random.seed(42)
mindspore.set_seed(42)
mindspore.context.set_context(device_target="GPU", pynative_synchronize=True)

int8_model = Linear8bitLt(4, 4, has_fp16_weights=False)
int8_model.quant()

input_data = Tensor(np.random.randn(1, 4).astype(np.float16))
int8_output = int8_model(input_data)

print(int8_output)
