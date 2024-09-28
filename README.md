# MindBNB

    quantization for mindnlp，本repo修改自经典的量化库bitsandbytes https://github.com/bitsandbytes-foundation/bitsandbytes，对齐mindspore生态进行了移植，在layer层面继承mindspore NLP中的mindnlp.core.nn.Linear实现了8bit量化层Linear8bitLt。

## Setup

安装最新的 mindnlp 版本

```
git clone https://github.com/mindspore-lab/mindnlp.git
cd mindnlp
bash scripts/build_and_reinstall.sh

```

安装依赖并编译 cuda 算子

bash /path/to/mindbnb/scripts/build.sh

## Start with quant_8bit

在 integrations.quantization_bnb_8bit 中实现了 quant_8bit 方法，在加载完模型后使用该接口对权重进行 8bit 量化，如下：

```
import sys
import os
import mindspore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integrations.quantization_bnb_8bit import quant_8bit
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

mindspore.set_context(device_target="GPU")

tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b")
model.set_train(False)

quant_8bit(model)

inputs = tokenizer("My favorite food is", return_tensors="ms")
output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
output_str = tokenizer.batch_decode(output_ids)[0]

print(output_str)
```

同时在 mindbnb/tests 中提供了三种层级（算子、layer、模型）的测试程序
