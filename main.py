from convolution_lstm import ConvLSTM
import torch
from torch.autograd import Variable

input = Variable(torch.rand(3, 3, 100, 400)).cuda()

model = ConvLSTM(3, [16, 2], 7, bias=True)
model.cuda()

for i in range(50):
    model.zero_grad()
    model.forward(input)
