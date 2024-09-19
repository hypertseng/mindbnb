# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained(
    "Rocketknight1/falcon-rw-1b", load_in_8bit=True
)

# for name, param in model.named_parameters():
#     print(name)
#     print(param)
inputs = tokenizer("My favorite food is", return_tensors="pt").to(0)
output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
output_str = tokenizer.batch_decode(output_ids)[0]

print(output_ids)
print(output_str)
