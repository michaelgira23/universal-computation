import tensorflow_datasets
import torch

from universal_computation.datasets.dataset import Dataset
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE

class AntiBiasDataset(Dataset):
    def __init__(self, device,vocab_size,batch_size, eval_batch_size,data_dir = "../unprejudiced_dataset/",model_name = "gpt2",input_max_dim = 50,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.device = device
        self.vocab_size = vocab_size

        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        file_train = open(data_dir+'train.txt', encoding='utf-8').readlines()
        file_test = open(data_dir+'test.txt', encoding='utf-8').readlines()

        tokenized_train = self.tokenizer(file_train)['input_ids']
        tokenized_test = self.tokenizer(file_test)['input_ids']

        padded_tokenized_train = [[tokenized_train[i][min(input_max_dim,len(tokenized_train[i])-1)]] + tokenized_train[i][:min(input_max_dim,len(tokenized_train[i])-1)] + [0]*max(0,input_max_dim+1-len(tokenized_train[i])) for i in range(len(tokenized_train))]
        padded_tokenized_test = [[tokenized_test[i][min(input_max_dim,len(tokenized_test[i])-1)]] + tokenized_test[i][:min(input_max_dim,len(tokenized_test[i])-1)] + [0]*max(0,input_max_dim+1-len(tokenized_test[i])) for i in range(len(tokenized_test))]

        # tokenized_train_y = [0] * vocab_size
        # tokenized_train_y[tokenized_train[:-1]] = 1
        # tokenized_test_y = [0] * vocab_size
        # tokenized_test_y[tokenized_test[:-1]] = 1

        #convert list to tensorflow Dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices(padded_tokenized_train)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(padded_tokenized_test)

        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=1024, reshuffle_each_iteration=True).batch(batch_size)
        self.test_dataset = self.test_dataset.batch(eval_batch_size)
        # self.test_dataset = self.test_dataset.shuffle(
        #     buffer_size=1024, reshuffle_each_iteration=True).batch(batch_size)

        self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


        self.train_enum = iter(tensorflow_datasets.as_numpy(self.train_dataset))
        self.test_enum = iter(tensorflow_datasets.as_numpy(self.test_dataset))

    
    def get_batch(self,train=True):
        if train:
            batch = next(self.train_enum, None)
            if batch is None:
                self.train_enum = iter(tensorflow_datasets.as_numpy(self.train_dataset))
                batch = next(self.train_enum)
        else:
            batch = next(self.test_enum, None)
            if batch is None:
                self.test_enum = iter(tensorflow_datasets.as_numpy(self.test_dataset))
                batch = next(self.test_enum)
        

        x = torch.from_numpy(batch[:,1:]).long()
        y_indices = batch[:,0]
        y = np.zeros((y_indices.size, self.vocab_size))
        y[np.arange(y_indices.size),y_indices] = 1
        y = torch.from_numpy(y).long()
        
        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x,y


        


        

