import torch
import torch as th
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F

class BehaviourNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,attention_dim):
        super(BehaviourNet, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.attention_dim = attention_dim

        self.encodeNet = nn.Linear(input_size, hidden_size)

        self.q = nn.Linear(self.hidden_size, self.attention_dim, bias=False)
        self.k = nn.Linear(self.hidden_size, self.attention_dim, bias=False)
        self.v = nn.Linear(self.hidden_size, self.attention_dim)

        self.rnn = nn.GRUCell(self.attention_dim, self.attention_dim)
        self.mlp = nn.Linear(self.attention_dim,self.output_size)

    def forward(self, obs, prev_hidden_state, hidden_state):
        _,h_length,_ = prev_hidden_state.shape
        h_c = F.relu(self.encodeNet(obs))
        h_c = h_c.view(-1,1, self.hidden_size)
        h_stack = th.cat((h_c,prev_hidden_state),dim=1)
        q = self.q(h_stack).reshape(-1,h_length+1,self.attention_dim)
        k = self.k(h_stack).reshape(-1,h_length+1,self.attention_dim)
        v = F.relu(self.v(h_stack)).reshape(-1,h_length+1,self.attention_dim)

        attention_scores = torch.matmul(q,k.transpose(-2,-1))/(np.sqrt(self.attention_dim))

        attention_weights = F.softmax(attention_scores,dim=-1)
        weighted_value = torch.matmul(attention_weights,v)
        attention_output = torch.sum(weighted_value,dim=1)

        new_hidden_state = self.rnn(attention_output,hidden_state)
        output = self.mlp(new_hidden_state)
        output = th.sigmoid(output)
        return output,new_hidden_state

if __name__ == '__main__':
    net = BehaviourNet(8,64,2,64)
    obs1 = th.zeros((1,1,8))
    prev_h = th.zeros((1,16,64))
    hidden = th.zeros((1,64))
    out,new_h = net(obs1,prev_h,hidden)
    print(out)
    print(new_h.shape)
    predicted = th.argmax(out,dim=1)
    print(predicted)
    one_hot = F.one_hot(predicted,num_classes=2)
    print(one_hot)


