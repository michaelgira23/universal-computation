import torch
from transformers import GPT2Tokenizer
from universal_computation.fpt_antibias import FPTAntiBias

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

while True:
    raw_text = input("Model prompt >>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input("Model prompt >>> ")

    # context_tokens = tokenizer(raw_text, return_tensors="pt")
    input_ids = torch.tensor(tokenizer.encode(raw_text, add_special_tokens=True))

    input_ids_tensor = torch.zeros([50], dtype=torch.int)

    for index, input_id in enumerate(input_ids):
        input_ids_tensor[index] = input_id

    print('raw text', raw_text, input_ids, input_ids.shape)
    print(input_ids_tensor, input_ids_tensor.shape)



    model = FPTAntiBias(input_max_dim = 50, device='cuda')

    hidden_states, output = model(input_ids_tensor)

    print('output', output, hidden_states)

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
