from convolution_lstm import ConvLSTM
import torch
from torch.autograd import Variable

""" Demo for the modified code """

input = Variable(torch.rand(3, 3, 100, 400)).cuda()

model = ConvLSTM(3, [64, 1], 7, bias=True)
model.cuda()

for i in range(1000):
    model.zero_grad()
    output, _ = model.forward(input)
    loss = output.mean()
    loss.backward()