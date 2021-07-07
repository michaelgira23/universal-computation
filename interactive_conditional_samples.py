import torch
from transformers import GPT2Tokenizer
from universal_computation.fpt_antibias import FPTAntiBias

experiment_params = dict(
    model_name='gpt2',
    input_max_dim=50,
    pretrained=True,
    freeze_trans=True,  # if False, we don't check arguments other than in and out
    freeze_linear=False,
    freeze_pos=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_out=False,
    linear_layer_sizes=None,  # not in paper, but can specify layer sizes for an MLP,
    # ex. [32, 32] creates a 2-layer MLP with dimension 32
    out_layer_sizes=None,
    learning_rate=1e-3,
    batch_size=2,
    dropout=0.1,
    orth_gain=1.41,
    position_ids=None,
    return_last_only=True,
)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

device = 'cuda'

while True:
    raw_text = input("Model prompt >>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input("Model prompt >>> ")

    # context_tokens = tokenizer(raw_text, return_tensors="pt")
    input_ids = torch.tensor(tokenizer.encode(raw_text, add_special_tokens=True))

    input_ids_tensor = torch.zeros([50], dtype=torch.int).to(device)

    for index, input_id in enumerate(input_ids):
        input_ids_tensor[index] = input_id

    print('raw text', raw_text, input_ids, input_ids.shape)
    print(input_ids_tensor, input_ids_tensor.shape)

    # model = FPTAntiBias(input_max_dim = 50, device='cuda')
    model = FPTAntiBias(
        input_max_dim=experiment_params['input_max_dim'],
        model_name=experiment_params['model_name'],
        pretrained=experiment_params['pretrained'],
        return_last_only=experiment_params['return_last_only'],
        linear_layer_sizes=experiment_params['linear_layer_sizes'],
        out_layer_sizes=experiment_params['out_layer_sizes'],
        freeze_trans=experiment_params['freeze_trans'],
        freeze_linear=experiment_params['freeze_linear'],
        freeze_pos=experiment_params['freeze_pos'],
        freeze_ln=experiment_params['freeze_ln'],
        freeze_attn=experiment_params['freeze_attn'],
        freeze_ff=experiment_params['freeze_ff'],
        freeze_out=experiment_params['freeze_out'],
        position_ids=experiment_params['position_ids'],
        dropout=experiment_params['dropout'],
        orth_gain=experiment_params['orth_gain'],
        device=device,
        stereo_set=True
    ).to(device)

    # FPTAntiBias(
    #     input_max_dim=input_max_dim,
    #     model_name=model_name,
    #     pretrained=kwargs.get('pretrained', True),
    #     return_last_only=kwargs.get('return_last_only', True),
    #     linear_layer_sizes=kwargs.get('linear_layer_sizes', None),
    #     out_layer_sizes=kwargs.get('out_layer_sizes', None),
    #     freeze_trans=kwargs.get('freeze_trans', True),
    #     freeze_linear=kwargs.get('freeze_linear', False),
    #     freeze_pos=kwargs.get('freeze_pos', False),
    #     freeze_ln=kwargs.get('freeze_ln', False),
    #     freeze_attn=kwargs.get('freeze_attn', True),
    #     freeze_ff=kwargs.get('freeze_ff', True),
    #     freeze_out=kwargs.get('freeze_out', False),
    #     position_ids = kwargs.get('position_ids',None),
    #     dropout=kwargs['dropout'],
    #     orth_gain=kwargs['orth_gain'],
    #     device = device,
    #     vocab_size = vocab_size,
    # )

    output = model(input_ids_tensor)
    print('output', output)

    # decoded_output = tokenizer.decode(output)

    # context_tokens = enc.encode(raw_text)
    # generated = 0
    # for _ in range(nsamples // batch_size):
    #     out = sess.run(output, feed_dict={
    #         context: [context_tokens for _ in range(batch_size)]
    #     })[:, len(context_tokens):]
    #     for i in range(batch_size):
    #         generated += 1
    #         text = enc.decode(out[i])
    #         print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
    #         print(text)
    print("=" * 80)
