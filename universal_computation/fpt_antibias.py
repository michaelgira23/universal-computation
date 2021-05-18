import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

class FPTAntiBias(nn.Module):

    def __init__(
            self,
            input_max_dim, 
            device,
            model_name='gpt2',
            pretrained = True,
            return_last_only=True, 
            linear_layer_sizes=None, 
            out_layer_sizes = None,
            freeze_trans=True,
            freeze_linear=False,
            freeze_pos=False,
            freeze_ln=False,
            freeze_attn=True,
            freeze_ff=True,
            freeze_out=False,
            position_ids = None, 
            dropout=0.1,
            orth_gain=1.41,
    ):
        super().__init__()

        self.input_max_dim = input_max_dim
        self.model_name = model_name
        self.return_last_only = return_last_only

        self.linear_layer_sizes = [] if linear_layer_sizes is None else linear_layer_sizes
        self.out_layer_sizes = [] if out_layer_sizes is None else out_layer_sizes
        self.dropout = dropout

        if 'gpt' in model_name:
            assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'] 

            from transformers import GPT2Model 

            pretrained_transformer = GPT2Model.from_pretrained(model_name)
            if pretrained:
                self.transformer = pretrained_transformer
            else:
                self.transformer = GPT2Model(pretrained_transformer.config)

            if model_name == 'gpt2':
                embedding_size = 768
            elif model_name == 'gpt2-medium':
                embedding_size = 1024
            elif model_name == 'gpt2-large':
                embedding_size = 1280
            elif model_name == 'gpt2-xl':
                embedding_size = 1600

        else:
            raise NotImplementedError('model_name not implemented')
        
        # TODO: Embedding layer
        self.wte = pretrained_transformer.wte #token embedding layer,
        self.wpe = pretrained_transformer.wpe #positional embedding layer,

        # linear layer between transformer and embedding layers
        linear_layer = []
        last_output_size = embedding_size
        for size in self.linear_layer_sizes:
            layer = nn.Linear(last_output_size, size)
            if orth_gain is not None:
                torch.nn.init.orthogonal_(layer.weight, gain=orth_gain)
            layer.bias.data.zero_()

            linear_layer.append(layer)
            linear_layer.append(nn.ReLU())
            linear_layer.append(nn.Dropout(dropout))
            last_output_size = size

        final_linear = nn.Linear(last_output_size, embedding_size) 
        if orth_gain is not None:
            torch.nn.init.orthogonal_(final_linear.weight, gain=orth_gain)
        final_linear.bias.data.zero_()

        linear_layer.append(final_linear)
        linear_layer.append(nn.Dropout(dropout))

        self.linear_net = nn.Sequential(*linear_layer)

        out_layers = []
        last_output_size = embedding_size
        for size in self.out_layer_sizes:
            out_layers.append(nn.Linear(last_output_size, size))
            out_layers.append(nn.ReLU())
            out_layers.append(nn.Dropout(dropout))
            last_output_size = size
        out_layers.append(nn.Linear(last_output_size, embedding_size))
        self.out_net = nn.Sequential(*out_layers)

        if freeze_trans:
            for name, p in self.transformer.named_parameters():
                name = name.lower()
                if 'ln' in name:
                    p.requires_grad = not freeze_ln
                elif 'wpe' in name:
                    p.requires_grad = not freeze_pos
                elif 'mlp' in name:
                    p.requires_grad = not freeze_ff
                elif 'attn' in name:
                    p.requires_grad = not freeze_attn
                else:
                    p.requires_grad = False
        if freeze_linear: #TODO: seperate linear layer and embedding layer
            for p in self.linear_net.parameters(): 
                p.requires_grad = False
        if freeze_out:
            for p in self.out_net.parameters():
                p.requires_grad = False

    def forward(self,x,output_attentions=False):

        # if position_ids is None:
        #     # position_ids = torch.from_numpy(np.arange(0,input_max_dim,dtype=int)[np.newaxis,:]).long()
        #     position_ids = torch.as_tensor(np.arange(0,self.input_max_dim,dtype=np.int_), dtype=torch.int64).to
        #     # position_ids = torch.range(0, 15,dtype=torch.int64)[None,:]
        # self.position_ids = position_ids

        # token embedding + positional embedding 
        # x = self.wte(x, mode="embedding") + self.position_embeds
        # x = self.wte(x) + self.wpe(position_ids)
        device = x.device
        batch_size = x.shape[0]
        position_list = [[*range(int(self.input_max_dim))]] * batch_size
        position_ids = torch.Tensor(position_list).to(device).long()

        x = self.wte(x) + self.wpe(position_ids)
        y= x[:,-1]

        # pass throught linear layers
        x = self.linear_net(x)

        transformer_outputs = self.transformer(
            inputs_embeds=x, 
            return_dict=True,
            output_attentions=output_attentions,
        )
        x = transformer_outputs.last_hidden_state

        # if self.return_last_only:
        #     x = x[:,-ratio:]

        x = self.out_net(x)

        if output_attentions:
            return x, y,transformer_outputs.attentions
        else:
            return x, y
