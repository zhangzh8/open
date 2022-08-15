import torch
from pytorch_transformers import BertTokenizer

model_name = 'bert-base-chinese'


samples = ['中国']
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenized_text = [tokenizer.tokenize(i) for i in samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_ids = torch.LongTensor(input_ids)
print(input_ids)

from pytorch_transformers import BertForMaskedLM

# 读取预训练模型
model = BertForMaskedLM.from_pretrained(model_name, cache_dir="./")
model.eval()

outputs = model(input_ids)
prediction_scores = outputs[0]
prediction_scores.shape
import numpy as np

sample = prediction_scores[0].detach().numpy()
pred = np.argmax(sample, axis=1)

print(len(tokenizer.convert_ids_to_tokens(pred)))

