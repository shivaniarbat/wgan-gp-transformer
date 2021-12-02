import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, TransformerDecoder, TransformerDecoderLayer
from torch.nn.modules.normalization import LayerNorm
from .model import Transformer_EncoderDecoder_Seq2Seq
from quaesita.PositionalEncoding import PositionalEncoding

torch.manual_seed(0)

class VanillaTransformerGenerator(nn.Module):
    """ Implements Vanilla Trasnformer Generator.

    The original Transformer Encoder-Decoder Model. The training process however differes
    in our work. Refer utils.utils.* for training procedure. This work is derived from pytorch
    implementation of Transformer Encoder-Decoder class. This is multi-step regression model.

    Arguments:
        model_params(dictionary): Refer example below -
            seq2seq_model_params = { 'd_model': 512, 'nhead': 8,'dropout': 0.1,'num_of_enc_layers': 6,'num_of_dec_layers': 6,'input_sequence_length': 10,'forecasting_step': 2}
    """
    def __init__(self, model_params):
        super(VanillaTransformerGenerator, self).__init__()
        self.model = Transformer_EncoderDecoder_Seq2Seq(**model_params)

    def forward(self, src, tgt, tgt_mask):
        output = model(src, tgt, tgt_mask)
        return self.model

class SequenceCritic(nn.Module):
    """ Implements Critic also can be referred as Discriminator. 

    Four Layer Multi Layer perceptron critic similar to original WGAN paper. https://arxiv.org/abs/1701.07875

    Arguments:
        model_params(dictionary): model dimensions as model parameters.
                                    model_params = { 'd_model' : 512, 'activation_fn' : 'ReLU'/'LeakyReLU'/ 'tanh' / 'Swish' }
    """
    def __init__(self, model_params):
        super(SequenceCritic, self).__init__()

        critic_model = nn.Sequential(
            nn.Linear(1, model_params['d_model']),
            nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(model_params['d_model'], model_params['d_model'] * 2),
            nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(model_params['d_model'] * 2, 1),
            nn.Tanh(),
            # nn.Sigmoid(),
        )

        self.critic_model = critic_model

    def forward(self, x):
        output = self.critic_model(x)
        return output
