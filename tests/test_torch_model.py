import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b", load_in_8bit=True)

# for name, param in model.named_parameters():
#     print(name)
#     print(param)
inputs = tokenizer("My favorite food is", return_tensors="pt").to(0)
output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
output_str = tokenizer.batch_decode(output_ids)[0]

print(output_ids)
print(output_str)