import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, TransformerDecoder, TransformerDecoderLayer
from torch.nn.modules.normalization import LayerNorm
# from torch.nn import xavier_uniform_
from quaesita.PositionalEncoding import PositionalEncoding

# sequence-to-sequence transformer
class VanillaTransformer_seq2seq(torch.nn.Module):

  def __init__(self, d_model = 512, nhead = 8, dropout = 0, num_of_layers = 6, input_sequence_length = 1,
  forecasting_step = 1):
    super(VanillaTransformer_seq2seq, self).__init__()
    self.model_name = "Sequence-to-Sequence Vanilla Transformer"
    self.d_model = d_model      # hidden dimension
    self.nhead = nhead          # parallel attention layers; multi-head-attention
    self.dropout = dropout
    self.num_of_layers = num_of_layers  # number of transformer encoder layer
    self.input_sequence_length = input_sequence_length 
    self.forecasting_step = forecasting_step

    # positional encoding
    self.positional_encoding = PositionalEncoding(self.d_model)

    # define single transformer encoder layer
    self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                 nhead = self.nhead)
    # layer normalization 
    self. encoder_norm = LayerNorm(self.d_model)

    # define transformer encoder 
    self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                                  num_layers = self.num_of_layers,
                                                  norm=self.encoder_norm)
    # linear layer
    self.decoder_layer = Linear(in_features = self.d_model,
                                out_features = 1) # self.output_sequence_length)
    
    # initialize weights
    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.decoder_layer.bias.data.zero_()
    self.decoder_layer.weight.data.uniform_(-initrange, initrange)

  def forward(self, x):

    pos_x = self.positional_encoding(x)

    transformer_encoded_x = self.transformer_encoder(pos_x)

    out = self.decoder_layer(transformer_encoded_x)

    out = out.squeeze(-1)
    
    return out

# one-step-ahead Vanilla Transformer model
class VanillaTransformeStepAhead(torch.nn.Module):
    def __init__(self,  d_model= 512, nhead = 8, num_of_layers = 6, dropout = 0.0,
    input_sequence_length = 1, forecasting_step = 1):
        super(VanillaTransformeStepAhead, self).__init__()
        self.model_name = 'Vanilla Transformer Step Ahead'
        self.d_model = d_model
        self.nhead = nhead
        self.num_of_layers = num_of_layers
        self.dropout = dropout
        self.input_sequence_length = input_sequence_length
        self.forecasting_step = forecasting_step

        # encoding layer
        self.positional_encoding = PositionalEncoding(self.d_model)

        # define single transformer encoder layer
        self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead = self.nhead)

        # layer normalization 
        self.encoder_norm = LayerNorm(self.d_model)
        
        # define transformer encoder 
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                                      num_layers = self.num_of_layers,
                                                      norm = self.encoder_norm)
        # linear layer
        self.decoder_layer = Linear(in_features = self.d_model,
                                    out_features = 1)
        
        # linear layer to map input_sequence features to forecasting_step_ahead number of features
        self.final_decoder_layer = Linear(in_features = self.input_sequence_length,
                                    out_features = self.forecasting_step)

        # initialize weights
        self.init_weights()
        
    def init_weights(self):
      initrange = 0.1
      self.decoder_layer.bias.data.zero_()
      self.decoder_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self,x):

      batch_size, in_seq_length, feature_size = x.shape

      pos_x = self.positional_encoding(x)
      transformer_encoded_x = self.transformer_encoder(pos_x)
      output = self.decoder_layer(transformer_encoded_x)

      # transform the input to predict steps ahead output
      output = output.permute(0,2,1).contiguous().view(-1, self.input_sequence_length)
      output = self.final_decoder_layer(output)
      output = output.view(-1, self.forecasting_step)
      return output

# sequence-to-sequence transformer with more linear layers
class Seq2seq_Transformer_Stacked_Linear_Layer(torch.nn.Module):

  def __init__(self, d_model = 512, nhead = 8, dropout = 0, num_of_layers = 6, input_sequence_length = 1,
  forecasting_step = 1):
    super(Seq2seq_Transformer_Stacked_Linear_Layer, self).__init__()
    self.model_name = "Seq2Seq Transformer with Linear decoder"
    self.d_model = d_model      # hidden dimension
    self.nhead = nhead          # parallel attention layers; multi-head-attention
    self.dropout = dropout
    self.num_of_layers = num_of_layers  # number of transformer encoder layer
    self.input_sequence_length = input_sequence_length 
    self.forecasting_step = forecasting_step

    # positional encoding
    self.positional_encoding = PositionalEncoding(self.d_model)

    # define single transformer encoder layer
    self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                 nhead = self.nhead)
    # layer normalization 
    self. encoder_norm = LayerNorm(self.d_model)

    # define transformer encoder 
    self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                                  num_layers = self.num_of_layers,
                                                  norm=self.encoder_norm)
    # decoder linear layer one
    self.decoder_layer1 = Linear(in_features = self.d_model,
                                out_features = 256)

    # decoder linear layer two
    self.decoder_layer2 = Linear(in_features = 256,
                                out_features = 128)
    
    # decoder linear layer three
    self.decoder_layer3 = Linear(in_features = 128,
                                out_features = 64)
    
    # decoder linear layer four
    self.decoder_layer4 = Linear(in_features = 64,
                                out_features = 1)

    # initialize weights
    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.decoder_layer1.bias.data.zero_()
    self.decoder_layer1.weight.data.uniform_(-initrange, initrange)

    self.decoder_layer2.bias.data.zero_()
    self.decoder_layer2.weight.data.uniform_(-initrange, initrange)

    self.decoder_layer3.bias.data.zero_()
    self.decoder_layer3.weight.data.uniform_(-initrange, initrange)

    self.decoder_layer4.bias.data.zero_()
    self.decoder_layer4.weight.data.uniform_(-initrange, initrange)

  def forward(self, x):

    # encoding
    pos_x = self.positional_encoding(x)

    # tranformer encoder
    transformer_encoded_x = self.transformer_encoder(pos_x)

    # stacked linear decoder
    out = self.decoder_layer1(transformer_encoded_x)
    out = self.decoder_layer2(out)
    out = self.decoder_layer3(out)
    out = self.decoder_layer4(out)

    out = out.squeeze(-1)
    
    return out

# sequence-to-sequence original transformer encoder-decoder architecture 
class Transformer_EncoderDecoder_Seq2Seq(torch.nn.Module):

  def __init__(self, d_model = 512, nhead = 8, dropout = 0.1, num_of_enc_layers = 6, num_of_dec_layers = 6,input_sequence_length = 1,
  forecasting_step = 1):
    super(Transformer_EncoderDecoder_Seq2Seq, self).__init__()
    self.model_name = "Sequence-to-Sequence Transformer Encoder Decoder"
    self.d_model = d_model      # hidden dimension
    self.nhead = nhead          # parallel attention layers; multi-head-attention
    self.dropout = dropout
    self.num_of_enc_layers = num_of_enc_layers  # number of transformer encoder layer
    self.num_of_dec_layers = num_of_dec_layers  # number of transformer decoder layer
    self.input_sequence_length = input_sequence_length 
    self.forecasting_step = forecasting_step

    # positional encoding
    self.positional_encoding = PositionalEncoding(self.d_model)

    # define single transformer encoder layer
    self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                 nhead = self.nhead,
                                                 dropout = self.dropout)
    # layer normalization 
    self. encoder_norm = LayerNorm(self.d_model)

    # define transformer encoder 
    self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                                  num_layers = self.num_of_enc_layers,
                                                  norm=self.encoder_norm)

    # define single transformer decoder layer 
    self.decoder_layer = TransformerDecoderLayer(d_model=self.d_model,
                                                nhead=self.nhead,
                                                dropout = self.dropout)
    # layer normalization 
    self.decoder_norm = LayerNorm(self.d_model)

    # define tranformer decoder
    self.decoder = TransformerDecoder(decoder_layer=self.decoder_layer,
                                                num_layers = self.num_of_dec_layers,
                                                norm=self.decoder_norm)

    # linear layer
    self.out = Linear(in_features = self.d_model,
                                out_features = 1) 

    # actiavtion layer
    self.tanh = torch.nn.Tanh()

    # initialize weights
    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.out.bias.data.zero_()
    self.out.weight.data.uniform_(-initrange, initrange)
                                        

    self._reset_parameters()

  def forward(self, src, tgt, tgt_mask):

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != tgt.size(2):
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src = self.positional_encoding(src))        # <---- encoder 
        output = self.decoder(tgt = self.positional_encoding(tgt), memory = memory, tgt_mask=tgt_mask) # <---- decoder 
        output = self.out(output)     # <---- linear layer 
        output = self.tanh(output)    # <---- activation layer

        output = output.squeeze(-1)
        return output

  def _reset_parameters(self):
    r"""Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
  
class NaiveModel(nn.Module):
  """ NaiveModel
  This model returns same value without processing it. 
  """
  def __init__(self, input_sequence_length = 1, forecasting_step = 1):
    super(NaiveModel, self).__init__()

    self.input_sequence_length = input_sequence_length
    self.forecasting_step = forecasting_step
  
  def forward(x):
    return x[-forecasting_step:]