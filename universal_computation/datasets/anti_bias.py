import tensorflow_datasets
import torch

from universal_computation.datasets.dataset import Dataset
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE

class AntiBiasDataset(Dataset):
    def __init__(self, batch_size, data_dir = "../universal-computation/unprejudiced_dataset/",model_name = "gpt2",input_max_dim = 50,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader

        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        file_train = open(data_dir+'train.txt', encoding='utf-8').readlines()
        file_test = open(data_dir+'test.txt', encoding='utf-8').readlines()

        tokenized_train = self.tokenizer(file_train)['input_ids']
        tokenized_test = self.tokenizer(file_test)['input_ids']

        padded_tokenized_train = [[0]*max(0,input_max_dim-len(tokenized_train[i])) + tokenized_train[i][:min(input_max_dim,len(tokenized_train[i]))] for i in range(len(tokenized_train))]
        padded_tokenized_test = [[0]*max(0,input_max_dim-len(tokenized_test[i])) + tokenized_test[i][:min(input_max_dim,len(tokenized_test[i]))] for i in range(len(tokenized_test))]

        #convert list to tensorflow Dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices(padded_tokenized_train)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(padded_tokenized_test)

        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=1024, reshuffle_each_iteration=True).batch(batch_size)
        self.test_dataset = self.test_dataset.batch(batch_size)

        self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


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

        x = torch.from_numpy(batch).long()
        
        x = x.to(device=self.device)

        self._ind += 1

        return x


        


        

