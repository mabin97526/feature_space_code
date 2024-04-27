import torch as th
import torch.nn as nn
if __name__ == '__main__':
    rnn = nn.GRU(128,128,num_layers=1,batch_first=True)
    input = th.ones((64,128))
    hid = th.ones((64,128))
    print(rnn(input,hid))