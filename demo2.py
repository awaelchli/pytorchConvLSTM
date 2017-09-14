from original.convolution_lstm import ConvLSTM
import torch
from torch.autograd import Variable

""" Demo for the original code from automan000 """

input = Variable(torch.rand(3, 3, 100, 400)).cuda()

model = ConvLSTM(3, [64, 2], 7, step=1, effective_step=[1], bias=True)
model.cuda()

for i in range(1000):
    model.zero_grad()
    output, _ = model.forward(input)
    print(output)
    loss = output.mean()
    loss.backward()