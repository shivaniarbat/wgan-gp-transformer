from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np

class timeseriesDataset(Dataset):
  def __init__(self, data, window_size=10, target_stride=10, batch_size=64):
    self.data = self._create_input_output_sequences(data, window_size, target_stride)
    self.window_size = window_size
    self.target_stride = target_stride
    self.batch_size = batch_size
    self.masks = self._generate_square_subsequent_mask(target_stride) # <------- generate target mask
    self.shape = self.__getshape__()
    self.size = self.__getsize__()

  def __getitem__(self, index):
    [x,y] = self.data[index]
    sample = (torch.Tensor(x).unsqueeze(-1),torch.Tensor(y).unsqueeze(-1), self.masks)
    return sample
 
  def __len__(self):    
    return len(self.data) 
    
  def __getshape__(self):
    return (self.__len__(), self.__getitem__(0)[0].shape)
    
  def __getsize__(self):
    return (self.__len__())

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
  
  def _create_input_output_sequences(self, data, window_size, target_stride):
    r"""Returns input output sequences for the provided window size and target stride
         To give an example -> 
         data = [1,2,3,....,99,100]
         window = 3
         target = 3
         output = [[1,2,3],[4,5,6],
                   [4,5,6],[7,8,9],...
                  ]
    """
    total_length = window_size + target_stride

    inout_sequences = []

    for idx in range(len(data) - window_size - target_stride):
      x = torch.reshape(data[idx:idx+window_size],(-1,))
      y = torch.reshape(data[idx+window_size:idx+window_size+target_stride:1],(-1,))
      inout_sequences.append([x.numpy(),y.numpy()])

    np_idx = np.arange(0,len(inout_sequences),target_stride)
    out = np.array(inout_sequences)[np_idx]
    return out

## THIS IS CORRECT BATCH CREATOR SEQUENCE ---> refer below
class timeseriesDatasetCreateBatch(Dataset):
    def __init__(self, data, window_size=10, target_stride=1, batch_size=64, flag=False): # TODO update this function to generate input for linear transformers
      self.data = torch.Tensor(data)
      self.window_size = window_size
      self.target_stride = target_stride
      
      self.batch_size = batch_size
      
      self.flag = flag
      self.input_output_sequences = self._create_inout_batch_sequences(self.data, self.batch_size, self.window_size, self.target_stride, self.flag)
      self.masks = self._generate_square_subsequent_mask(target_stride) # <------- generate target mask

      self.shape = self.__getshape__()
      self.size = self.__getsize__()

    def __getitem__(self, index):
      (X,Y) = self.input_output_sequences
      x = X[index]
      y = Y[index]
      sample = (x,y, self.masks)
      
      return sample
  
    def __len__(self):    
      return len(self.data) -  self.window_size - self.target_stride
      
    def __getshape__(self):
      return (self.__len__(), self.__getitem__(0)[0].shape)
      
    def __getsize__(self):
      return (self.__len__())

    def _generate_square_subsequent_mask(self,window_size):
          mask = (torch.triu(torch.ones(window_size, window_size)) == 1).transpose(0, 1)
          mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
          return mask
    
    def _create_inout_batch_sequences(self, data, batch_size, window_size, target_stride, flag):
      total_len = window_size + target_stride

      x_src = []
      y_tar = []

      for index in range(0,len(data) - total_len + 1, 1):
        _data = data[index : index + total_len].numpy()
        x = torch.reshape(torch.FloatTensor(_data[:window_size]),(window_size,1))
        if flag: 
          # this flag set to True when we generate input ouput as below:
          # data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
          # x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
          # y = [1, 2, 3, 4, 5, 6, 7, 8, 9] # default target_stride is = 1
          y = torch.reshape(torch.FloatTensor(_data[-window_size:]),(window_size,1))
        else:
          y = torch.reshape(torch.FloatTensor(_data[-target_stride:]),(-1,))

        x_src.append(x)
        y_tar.append(y)

      # to create batches
      x_batches = []
      y_batches = []

      i = 0
      remaining = len(x_src)
      while(i < len(x_src)):
        # print("in for look:",i)
        if remaining <= batch_size: 
          # print('FOUND IT', remaining)
          batch_size = remaining

        x_temp = torch.stack([x_src[i+j] for j in range(batch_size)],1)    # this tensors are stacked with dim = 1 to match the input dimension requirement of transformer tensor
        y_temp = torch.stack([y_tar[i+j] for j in range(batch_size)],1)    
          
        x_batches.append(x_temp)
        y_batches.append(y_temp)
        i += batch_size
        remaining -= batch_size
        
        # sample = (x_batches, y_batches)
      return x_batches, y_batches