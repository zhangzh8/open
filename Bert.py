import torch

from dp4.transform import tokenizer, model_name

with open('./neg.txt', 'r', encoding='utf-8') as f:
    neg_data = f.read()
with open('./pos.txt', 'r', encoding='utf-8') as f:
    pos_data = f.read()

neg_datalist = neg_data.split('\n')
pos_datalist = pos_data.split('\n')


import numpy as np

dataset = np.array(pos_datalist + neg_datalist)
labels = np.array([1] * len(pos_datalist) + [0] * len(neg_datalist))
print(dataset)
np.random.seed(10)
mix_index = np.random.choice(10000, 10000)
dataset = dataset[mix_index]
labels = labels[mix_index]


TRAINSET_SIZE = 8000
EVALSET_SIZE = 2000

train_samples = dataset[:TRAINSET_SIZE]  # 2500 条数据
train_labels = labels[:TRAINSET_SIZE]
eval_samples = dataset[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]  # 500 条数据
eval_labels = labels[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]



def get_dummies(l, size=2):
    res = list()
    for i in l:
        tmp = [0] * size
        tmp[i] = 1
        res.append(tmp)
    return res


from torch.utils.data import DataLoader, TensorDataset

tokenized_text = [tokenizer.tokenize(i) for i in train_samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_labels = get_dummies(train_labels)  # 使用 get_dummies 函数转换标签

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) <= 100:
        input_ids[j].extend([0] * (100 - len(i)))
    else:
        del input_ids[j][100: len(i)]

train_set = TensorDataset(torch.LongTensor(input_ids),
                          torch.FloatTensor(input_labels))
train_loader = DataLoader(dataset=train_set,
                          batch_size=1,
                          shuffle=True)

tokenized_text = [tokenizer.tokenize(i) for i in eval_samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_labels = eval_labels

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) <= 100:
        input_ids[j].extend([0] * (100 - len(i)))
    else:
        del input_ids[j][100: len(i)]

eval_set = TensorDataset(torch.LongTensor(input_ids),
                         torch.FloatTensor(input_labels))
eval_loader = DataLoader(dataset=eval_set,
                         batch_size=1,
                         shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel


class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BertModel.from_pretrained(model_name, cache_dir="./")
        self.model.to(device)
        self.dropout = nn.Dropout(0.08)
        self.l1 = nn.Linear(768, 2)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[1]  # 取池化后的结果 batch * 768
        x = x.view(-1, 768)
        x = self.dropout(x)
        x = self.l1(x)
        return x


from torch import optim

cls = fn_cls()
cls.to(device)
cls.train()

criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = optim.Adam(cls.parameters(), lr=0.00001)

def predict(logits):
    res = torch.argmax(logits, 1)
    return res


from torch.autograd import Variable
import time
from tqdm import tqdm_notebook as tqdm
pre = time.time()

accumulation_steps = 32
epoch = 10

for i in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target.view(-1, 2)).to(device)

        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)

        output = cls(data, attention_mask=mask)
        pred = predict(output)

        loss = criterion(sigmoid(output).view(-1, 2), target)
        # 梯度积累
        loss = loss / accumulation_steps
        loss.backward()

        if ((batch_idx + 1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
        if ((batch_idx + 1) % accumulation_steps) == 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                i + 1, batch_idx, len(train_loader), 100. *
                batch_idx / len(train_loader), loss.item()
            ))

    cls.eval()

    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
        data = data.to(device)
        target = target.long().to(device)

        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)

        output = cls(data, attention_mask=mask)
        pred = predict(output)

        correct += (pred == target).sum().item()
        total += len(data)


    print('epoch',i)
    print('正确分类的样本数：{}，样本总数：{}，准确率：{:.2f}%'.format(
        correct, total, 100. * correct / total))
    print('训练时间：', time.time() - pre)
    f = open(r"dp4result.txt", 'a')
    f.write('epoch:')
    f.write(str(epoch))
    f.write('acc:')
    f.write(str(100. * correct / total))
    f.write('\n')



