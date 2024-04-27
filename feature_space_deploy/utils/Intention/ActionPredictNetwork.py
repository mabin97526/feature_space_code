import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size+2, hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.shape[0]
        input = F.relu(self.linear(input))
        #input 64*1*128 hid: 1*64*128
        output, new_hidden = self.rnn(input,hidden)
        new_hidden_out = self.out(output.reshape(batch_size,-1))
        #new_hidden_out = F.softmax(new_hidden_out, dim=-1)
        return output, new_hidden, new_hidden_out

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, action_shape,state_shape, num_layers, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size*2, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)
        self.outaction1 = nn.Linear(hidden_size, action_shape)
        self.outaction2 = nn.Linear(hidden_size, action_shape)
        self.outstate = nn.Linear(hidden_size, state_shape)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        batch_size = encoded_input.shape[0]
        #before : 64*128 after:1*64*128
        hidden = hidden.unsqueeze(0).contiguous()

        encoded_input = F.relu(self.linear(encoded_input)).unsqueeze(1)
        decoded_output, hidden = self.rnn(encoded_input, hidden)
        decoded_output = self.tanh(decoded_output)
        decoded_output = self.dropout(decoded_output)
        p_state = self.outstate(decoded_output)
        p_action1 = self.outaction1(decoded_output)
        p_action2 = self.outaction2(decoded_output)
        #p_action = F.softmax(p_action,dim=-1)
        return p_state,p_action1,p_action2, hidden

class Action_Latent_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 action_shape,state_shape, dropout=0.5):
        super(Action_Latent_Decoder, self).__init__()

        self.decoder = RNNDecoder(input_size, hidden_size,action_shape,state_shape, num_layers, dropout)

    def forward(self, cur_h, pre_latent, hidden):

        #batch_size, hidden_dim = cur_h.shape
        decoder_input = th.cat([cur_h, pre_latent],dim=-1)

        state,action1,action2,hidden = self.decoder(decoder_input,hidden)
        return state,action1,action2, hidden
