import tensorflow_datasets
import torch

from universal_computation.datasets.dataset import Dataset
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
AUTOTUNE = tf.data.experimental.AUTOTUNE

class AntiBiasDataset(Dataset):
    def __init__(self, batch_size, data_dir,model_name = "gpt2",input_max_dim = 50,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader

        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        sel_cols = ['label', 'sentence']
        col_defaults = [tf.int32, tf.string]
        # TODO: decide the data_dir and files under this directory (i.e., training, testing, evaluation)
        self.d_train_str = tf.data.experimental.make_csv_dataset(data_dir+'antibias_train.tsv',
                                               batch_size,
                                               column_defaults=col_defaults,
                                               select_columns=sel_cols,
                                               field_delim='\t',
                                               header=True,
                                               num_epochs=1)
        self.d_train_str = self.d_train_str.unbatch()
        self.d_test_str =  tf.data.experimental.make_csv_dataset(data_dir+'antibias_test.tsv',
                                               batch_size,
                                               column_defaults=col_defaults,
                                               select_columns=sel_cols,
                                               field_delim='\t',
                                               header=True,
                                               num_epochs=1)
        self.d_test_str = self.d_test_str.unbatch()


        # tokenize sentences
        def tokenize(d):
            return {
                'inputs': self.tokenizer(d['sentence'],return_tensors='pt')[:input_max_dim],
                'label': d['label']
            }

        train_dataset = self.d_train_str.map(tokenize, num_parallel_calls=AUTOTUNE)
        test_dataset = self.d_test_str.map(tokenize, num_parallel_calls=AUTOTUNE)

        max_shape = {'inputs': [input_max_dim], 'label': []}
        train_dataset = train_dataset.shuffle(
            buffer_size=1024, reshuffle_each_iteration=True).padded_batch(
                batch_size, padded_shapes=max_shape)
        test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=max_shape)

        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.train_enum = iter(tensorflow_datasets.as_numpy(self.train_dataset))
        self.test_enum = iter(tensorflow_datasets.as_numpy(self.test_dataset))

    
    def get_batch(self, batch_size=None, input_max_dim=50, model_name = "gpt2", train=True):
        if train:
            batch = next(self.train_enum, None)
            if batch is None:
                self.train_enum = iter(tensorflow_datasets.as_numpy(self.d_train))
                batch = next(self.train_enum)
        else:
            batch = next(self.test_enum, None)
            if batch is None:
                self.test_enum = iter(tensorflow_datasets.as_numpy(self.d_test))
                batch = next(self.test_enum)

        x, y = batch['inputs'], batch['label']
        x = torch.from_numpy(x).long()
        y = torch.from_numpy(y).long()

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y


        


        

