import numpy as np
import torch
import wandb

import argparse
from datetime import datetime
import random
import sys

from universal_computation.fpt_anibias import FPTAntiBias
from universal_computation.trainer import Trainer


def experiment(
        exp_name,
        exp_args,
        **kwargs
):

    """
    Preliminary checks
    """

    # Must be able to accumulate gradient if batch size is large
    assert 'batch_size' in kwargs
    assert kwargs['batch_size'] <= exp_args['gpu_batch_size'] or \
           kwargs['batch_size'] % exp_args['gpu_batch_size'] == 0

    """
    Create dataset, model, and trainer
    """

    task = kwargs['task']
    batch_size = kwargs['batch_size']
    device = exp_args['device']
    model_name = 
    input_max_dim = 



    return_last_only = True


    from universal_computation.datasets.anti_bias import AntiBiasDataset
    dataset = AntiBiasDataset(batch_size=batch_size,model_name = model_name, input_max_dim = input_max_dim, device=device)
    input_dim, output_dim = 15, 10
    use_embeddings = True

    experiment_type = #TODO
    ce_loss = #TODO
    loss_fn = #TODO
    accuracy_fn = #TODO
    



    model = FPTAntiBias(
        input_max_dim=input_max_dim,
        output_dim=output_dim,
        model_name=kwargs.get('model_name', 'gpt2'),
        pretrained=kwargs.get('pretrained', True),
        return_last_only=return_last_only,
        use_embeddings_for_in=use_embeddings,
        linear_layer_sizes=kwargs.get('linear_layer_sizes', None),
        out_layer_sizes=kwargs.get('out_layer_sizes', None),
        freeze_trans=kwargs.get('freeze_trans', True),
        freeze_in=kwargs.get('freeze_in', False),
        freeze_pos=kwargs.get('freeze_pos', False),
        freeze_ln=kwargs.get('freeze_ln', False),
        freeze_attn=kwargs.get('freeze_attn', True),
        freeze_ff=kwargs.get('freeze_ff', True),
        freeze_out=kwargs.get('freeze_out', False),
        position_ids = kwargs.get('position_ids',None),
        dropout=kwargs['dropout'],
        orth_gain=kwargs['orth_gain'],
    )
    model.to(device)

    gpu_batch_size = exp_args['gpu_batch_size']
    trainer = Trainer(
        model,
        dataset,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        steps_per_epoch=exp_args['steps_per_iter'],
        test_steps_per_epoch=exp_args['test_steps_per_iter'],
        learning_rate=kwargs['learning_rate'],
        batch_size=gpu_batch_size if batch_size > gpu_batch_size else batch_size,
        eval_batch_size=batch_size,
        grad_accumulate=batch_size // gpu_batch_size if batch_size > gpu_batch_size else 1,
    )

    """
    Set up logging
    """

    log_to_wandb = exp_args['log_to_wandb']
    save_models = exp_args['save_models']
    wandb_project = exp_args['wandb_project']

    short_name = str(random.randint(int(1e5), int(1e6) - 1))
    run_name = f'{exp_name}-{task}-{short_name}'

    if log_to_wandb:
        config = dict(
            short_name=short_name,
            run_name=run_name,
            **exp_args,
            **kwargs,
        )
        wandb.init(
            name=f'{exp_name}-{short_name}',
            group=f'{exp_name}-{task}',
            project=wandb_project,
            config=config,
        )
        wandb.watch(model)

    for t in range(exp_args['num_iters']):
        trainer.train_epoch()

        print('=' * 57)
        print(f'| Iteration {" " * 15} | {t+1:25} |')
        for k, v in trainer.diagnostics.items():
            print(f'| {k:25} | {v:25} |')

        if log_to_wandb:
            wandb.log(trainer.diagnostics)

        if save_models and ((t+1) % exp_args['save_models_every'] == 0 or
                            (t+1) == exp_args['num_iters']):
            with open(f'models/{run_name}.pt', 'wb') as f:
                state_dict = dict(model=model.state_dict(), optim=trainer.optim.state_dict())
                torch.save(state_dict, f)
            print(f'Saved model at {t+1} iters: {run_name}')


def run_experiment(
        exp_name,
        experiment_params,
):
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_iters', '-it', type=int, default=10000,
                        help='Number of iterations for trainer')
    parser.add_argument('--steps_per_iter', type=int, default=100,
                        help='Number of gradient steps per iteration')
    parser.add_argument('--test_steps_per_iter', type=int, default=25,
                        help='Number of test gradient steps per iteration')

    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False,
                        help='Whether or not to log to Weights and Biases')
    parser.add_argument('--note', '-n', type=str, default='',
                        help='An optional note to be logged to W&B')
    parser.add_argument('--wandb_project', type=str, default='my_project',
                        help='Project name for W&B')
    parser.add_argument('--include_date', type=bool, default=True,
                        help='Whether to include date in run name')

    parser.add_argument('--save_models', '-s', type=bool, default=False,
                        help='Whether or not to save the model files locally')
    parser.add_argument('--save_models_every', '-int', type=int, default=25,
                        help='How often to save models locally')

    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Which device for Pytorch to use')
    parser.add_argument('--gpu_batch_size', '-gbs', type=int, default=16,
                        help='Max batch size to put on GPU (used for gradient accumulation)')

    exp_args = parser.parse_args(sys.argv[1:])

    if exp_args.include_date:
        timestamp = datetime.now().strftime('%m-%d')
        exp_name = f'{timestamp}-{exp_name}'

    experiment_params['exp_name'] = exp_name
    experiment_params['exp_args'] = vars(exp_args)

    experiment(xp_name=exp_name, **experiment_params)
