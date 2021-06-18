import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from universal_computation.experiment_antibias import run_experiment

if __name__ == '__main__':

    experiment_name = 'fpt_antibias'

    experiment_params = dict(
        task='antibias',
        model_name='gpt2',
        input_max_dim = 50,
        pretrained=True,
        freeze_trans=True,  # if False, we don't check arguments other than in and out
        freeze_linear=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,
        linear_layer_sizes= None,  # not in paper, but can specify layer sizes for an MLP,
        out_layer_sizes=None,  # ex. [32, 32] creates a 2-layer MLP with dimension 32
        learning_rate=1e-3,
        batch_size=2,
        eval_batch_size = 8,
        dropout=0.1,
        orth_gain=1.41,
        position_ids = None,
        return_last_only = True,
        vocab_size = 50257,
    )

    run_experiment(experiment_name, experiment_params)
