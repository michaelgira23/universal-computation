import torch
import torch.nn as nn


class FPTAntiBias(nn.Module):

    def __init__(
            self,
            input_max_dim, #TODO: change the name of parmameters accordingly
            output_dim,
            model_name='gpt2',
            pretrained=False,
            return_last_only=True, #TODO: how to deal with this case? what means return_last_only
            use_embeddings_for_in=False,
            linear_layer_sizes=None, #TODO: change the name of parmameters accordingly
            out_layer_sizes=None,
            freeze_trans=True,
            freeze_in=False,
            freeze_pos=False,
            freeze_ln=False,
            freeze_attn=True,
            freeze_ff=True,
            freeze_out=False,
            position_ids = None, #TODO: how to set up the position id
            dropout=0.1,
            orth_gain=1.41,
    ):
        super().__init__()

        self.input_max_dim = input_max_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.return_last_only = return_last_only
        self.use_embeddings_for_in = use_embeddings_for_in

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
        self.wte = pretrained_transformer.wte#weights of token embedding, type: tensor
        self.wpe = pretrained_transformer.wpe #weights of positional embedding, type: tensor
        
        # if position_ids is None:
        #     position_ids = 

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
        out_layers.append(nn.Linear(last_output_size, output_dim))
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
        if freeze_in: #TODO: seperate linear layer and embedding layer
            for p in self.linear_net.parameters(): 
                p.requires_grad = False
        if freeze_out:
            for p in self.out_net.parameters():
                p.requires_grad = False

    # TODO: how do we want the format of input to be, currently assume it to input_ids of one tokenized sentences 
    def forward(self, x, output_attentions=False):

        #TODO: do we pass one sentence as input at one time (x is 1-d) or multiple sentences (x is 2-d) 
        # # reshape x
        # orig_dim = x.shape[-1]
        # if orig_dim > self.input_max_dim: # cut off the tail
        #     x = x[...,0:self.input_max_dim]
        # elif orig_dim < self.input_max_dim: # adding padding 
        #     shape = x.shape.as_list()
        #     shape[-1] = self.input_max_dim
        #     shape = tuple(shape)
        #     target_x = torch.zeros(shape)
        #     target_x[...,:orig_dim] = x
        #     x = target_x

        # token embedding 
        x=  self.emb_layer(x)
        x = self.linear_net(x)

        transformer_outputs = self.transformer(
            inputs_embeds=x, # that's how the embeded vector is passed into the embedding layer.
            return_dict=True,
            output_attentions=output_attentions,
        )
        x = transformer_outputs.last_hidden_state

        # if self.return_last_only:
        #     x = x[:,-ratio:]

        x = self.out_net(x)
        # if self.return_last_only and ratio > 1:
        #     x = x.reshape(x.shape[0], x.shape[1] // ratio, ratio * self.output_dim)

        if output_attentions:
            return x, transformer_outputs.attentions
        else:
            return x
