#!/usr/bin/env python2
import math
import json
import time
import sys
import argparse
import numpy as np
from numpy.lib.twodim_base import triu_indices_from
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils.utils import mean_absolute_percentage_error as MAPE
from utils.utils import plot_output, train_tf_enc_dec_gan, train_and_evaluate_tf_enc_dec_gan, test_model
from quaesita.pre_process_data import getSampledData
from quaesita.model import Transformer_EncoderDecoder_Seq2Seq, VanillaTransformer_seq2seq
from quaesita.TimeSeriesDataset import timeseriesDatasetCreateBatch 
from utils.loss.dilate_loss import DILATE_loss
from quaesita.transformerGANs import VanillaTransformerGenerator, SequenceCritic
from utils.optimizer import MADGRAD, improved_gradient_penalty
from utils.utils import train_tf_encdec, test_tf_enc_dec
from utils.utils import mean_absolute_percentage_error as MAPE
from utils.utils import root_mean_square_error as RMSE
from utils.utils import mean_absolute_scaled_error as MASE
import csv

torch.manual_seed(1)

EXP_FOLDER_PATH = 'cloudlabgpu1-output/TranImpWGAN/'

D_MODEL = int(sys.argv[1])
N_HEAD = int(sys.argv[2])
DROPOUT = float(sys.argv[3])

WINDOW_SIZE = sys.argv[4]
BATCH_SIZE = sys.argv[5]

DATASET_NAME = sys.argv[6]
GPU = sys.argv[7]

# read params path file for input data
"""
keys()
["google-1m", "azure-10m"]
"""
with open('data/workloads-params.json') as f: 
# with open('data/workloads-params.json') as f: 
    input_workload = json.load(f)

in_workload_key = str(DATASET_NAME)
print(in_workload_key)

time_interval = 30
event_type = 0
scheduling_class = 1
layers_num = 6


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#<-----------------------*-*-*-*------------------------>
# Training parameters 
#<-----------------------*-*-*-*------------------------>

WEIGHT_DECAY = 'DEFAULT' # weight decay for MADGRAD optimizer for Generator

lossG_type = 'mae'
criterionG = torch.nn.L1Loss()

# epochs 
epochs = 1000

one = torch.ones([])
one = one.to('cuda')
mone = one * -1

print(one, mone)

#<-----------------------*-*-*-*------------------------>

def loss_quantile(mu:Variable, labels:Variable, quantile:Variable):
    loss = 0
    for i in range(mu.shape[1]):
        mu_e = mu[:, i].to('cuda')
        labels_e = labels[:, i].to('cuda')

        I = (labels_e >= mu_e).float().to('cuda')
        each_loss = 2*(torch.sum(quantile*((labels_e -mu_e)*I)+ (1-quantile) *(mu_e- labels_e)*(1-I))).to('cuda')
        loss += each_loss.to('cuda')

    return loss

def train(model,
            discriminator,
            criterionG,
            optimizer_G,
            optimizer_D,
            adverserial_loss,
            train_dl,
            dataset_params,
            scaler):
    
    model.train()

    batch_size = dataset_params['batch_size']
    forecasting_step = dataset_params['target_stride']
    window_size = dataset_params['window_size']

    x_input = []
    truth = []
    predicted = []

    train_loss = 0
    trainD_loss = 0

    n = 0 

    for x, y, tgt_mask in train_dl:
        optimizer_G.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')

        # encode x
        enc_out = model.encoder(model.positional_encoding(x)).to('cuda')

        out = torch.clone(x[-1:])   # last bit from the input
        outputs = torch.clone(x[-forecasting_step:])

        # decode x
        for step in range(1, forecasting_step + 1, 1):
            mask = (torch.triu(torch.ones(1, 1)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            tgt_mask = mask.to('cuda')

            if step == 1:
                dec_in = x[-1:]
            else:
                dec_in = out
            
            dec_in_emb = model.positional_encoding(dec_in).to('cuda')

            out = model.out(model.decoder(dec_in_emb, enc_out, tgt_mask))
            
            outputs[step - 1:step:] = out 
            
        
        y = y.unsqueeze(-1)

        _fake_input = torch.cat((x, outputs), 0)
        fake_input = torch.stack([_fake_input.squeeze(-1)[i] for i in range(_fake_input.shape[0])],-1)  # fake_input dims => [batch, window_size, features] {LINEAR MODEL}
        fake_input = fake_input.unsqueeze(-1).to('cuda') 
        fake_input = autograd.Variable(fake_input)

        _real_input = torch.cat((x,y), 0)   
        real_input = torch.stack([_real_input.squeeze(-1)[i] for i in range(_real_input.shape[0])],-1)  # real_input dims => [batch, window_size, features] {LINEAR MODEL}
        real_input = real_input.unsqueeze(-1).to('cuda')   
        real_input = autograd.Variable(real_input)

        if adverserial_loss == 'improvedWGAN':

            for p in discriminator.parameters():
              p.requires_grad = True

            # discriminator update
            optimizer_D.zero_grad()
            d_real = discriminator(real_input)
            d_real = d_real.mean()
            d_real.backward(mone)

            d_fake = discriminator(fake_input.detach())
            d_fake = d_fake.mean()
            d_fake.backward(one)

            gradient_penalty = improved_gradient_penalty(discriminator, real_input, fake_input.detach())
            gradient_penalty.backward()

            loss_d = d_fake - d_real + gradient_penalty * 100 # LAMBDA = 0.1 
            optimizer_D.step()

            trainD_loss += (loss_d.item())

            # generator update
            for p in discriminator.parameters():
              p.requires_grad = False

            g_d_fake_input_loss = discriminator(fake_input)
            g_d_fake_input_loss = g_d_fake_input_loss.mean()
            loss = criterionG(outputs, y) + g_d_fake_input_loss
            loss.backward(one)
            optimizer_G.step()

            train_loss += (loss.item() * x.shape[0])
            n += x.shape[0]

        x = x.to('cpu') 
        y = y.to('cpu')
        outputs = outputs.detach().to('cpu')
        
        x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
        truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
        predicted.append(scaler.inverse_transform(np.reshape(np.array(outputs[0].view(-1).numpy()),(outputs.shape[1],1))))

    return train_loss/n, trainD_loss/n,  x_input, truth, predicted#, shape_l, temporal_l

def train_generator(model,
            discriminator,
            criterionG,
            optimizer_G,
            train_dl,
            dataset_params,
            scaler):
    
    model.train()

    batch_size = dataset_params['batch_size']
    forecasting_step = dataset_params['target_stride']
    window_size = dataset_params['window_size']

    x_input = []
    truth = []
    predicted = []

    train_loss = 0
    trainD_loss = 0

    n = 0 

    for x, y, tgt_mask in train_dl:
        optimizer_G.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')

         # Adversarial ground truths
        valid = torch.autograd.Variable(torch.cuda.FloatTensor(window_size + 1, batch_size,1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(torch.cuda.FloatTensor(window_size + 1, batch_size,1).fill_(0.0), requires_grad=False)
        
        # encode x
        enc_out = model.encoder(model.positional_encoding(x)).to('cuda')

        out = torch.clone(x[-1:])   # last bit from the input
        outputs = torch.clone(x[-forecasting_step:])

        # decode x
        for step in range(1, forecasting_step + 1, 1):
            mask = (torch.triu(torch.ones(1, 1)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            tgt_mask = mask.to('cuda')

            if step == 1:
                dec_in = x[-1:]
            else:
                dec_in = out
            
            dec_in_emb = model.positional_encoding(dec_in).to('cuda')

            out = model.out(model.decoder(dec_in_emb, enc_out, tgt_mask))
            
            outputs[step - 1:step:] = out 
            
        
        y = y.unsqueeze(-1)

        # generator update
        loss = criterionG(outputs, y)
        loss.backward()
        optimizer_G.step()

        train_loss += (loss.item() * x.shape[0])
        n += x.shape[0]

        x = x.to('cpu') 
        y = y.to('cpu')
        outputs = outputs.detach().to('cpu')
        
        x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
        truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
        predicted.append(scaler.inverse_transform(np.reshape(np.array(outputs[0].view(-1).numpy()),(outputs.shape[1],1))))
           
    return train_loss/n, trainD_loss/n,  x_input, truth, predicted

def call_main(_window_size, _batch_size,_train_data, _cross_val_data, _test_data, _gpu_name):

    x_input, truth , predicted = [],[],[]

    # output datastructure to csv
    results = np.empty([0,23],str)

    # for _window_size, _batch_size in experiment_params:
    seq2seq_dataset_params = {
        'window_size' : int(_window_size),   # input sequence length
        'target_stride' : 1,   # predict number of target steps
        'batch_size' : int(_batch_size),
        'flag'  :   False
    }
    print(seq2seq_dataset_params)

    #<-----SEQ2SEQ MODEL PARAMS & DATA (MODEL A)------>

    # datasets and dataloaders for MODEL A
    train_datasetA = timeseriesDatasetCreateBatch(_train_data, **seq2seq_dataset_params)  
    test_datasetA = timeseriesDatasetCreateBatch(_test_data, **seq2seq_dataset_params)  
    val_datasetA = timeseriesDatasetCreateBatch(_cross_val_data,  **seq2seq_dataset_params)

    __d_model = (int(seq2seq_dataset_params['window_size']) // 2) * 2
    seq2seq_modelG_params = {
        'd_model': D_MODEL,
        'nhead': N_HEAD,
        'dropout': DROPOUT,
        'num_of_enc_layers': 1,
        'num_of_dec_layers': 1,
        'input_sequence_length': int(seq2seq_dataset_params['window_size']),
        'forecasting_step': 1
    }

    print(seq2seq_modelG_params)

    modelD_params = {
        'd_model' : seq2seq_modelG_params['d_model'],
        'activation_fn' : 'LeakyReLU'
    }

    #<-----------------------*-*-*-*------------------------>
    # Generator Model definition
    #<-----------------------*-*-*-*------------------------>
    # DEFINE MODEL G
    modelG = Transformer_EncoderDecoder_Seq2Seq(**seq2seq_modelG_params)
    modelG.to('cuda')

    #<-----------------------*-*-*-*------------------------>
    # Critic Model definition
    #<-----------------------*-*-*-*------------------------>
    # DEFINE MODEL D
    modelD = SequenceCritic(modelD_params)
    modelD.to('cuda')

    step_num = len(train_datasetA) // int(_batch_size)
    lr = 0.01

    # optimizers for respective model
    optimizer_G = MADGRAD(modelG.parameters(), lr= lr)#, weight_decay = WEIGHT_DECAY)       # <---- weight decay here is default
    optimizer_D = MADGRAD(modelD.parameters(), lr= lr)                                      # <---- weight decay here is default

    start_time = time.time()                # start time for the entire loop

    train_start_time = time.time() - start_time
    # for e, epoch in enumerate(range(epochs)):
    for e in tqdm(range(epochs)):
        trainG_loss, trainD_loss,  x_input, truth, predicted = train(model = modelG, discriminator = modelD, 
                    criterionG = criterionG,
                    optimizer_G = optimizer_G,
                    optimizer_D = optimizer_D,
                    # G_scheduler = G_scheduler,
                    # D_scheduler = D_scheduler,
                    adverserial_loss = 'improvedWGAN',
                    train_dl = train_datasetA,
                    dataset_params = seq2seq_dataset_params,
                    scaler = scaler)
        # trainD_loss = 0
        if e % 100 == 0 or (e == epochs - 1):
            for i in range(len(truth)):
                truth[i] = truth[i].flatten()
                predicted[i] = predicted[i].flatten()
            truth = np.hstack(truth)
            predicted = np.hstack(predicted)
            mape_a = np.round(MAPE(truth, predicted),2)
            print("Epoch:{}  Train MAPE:{}   G Loss:{}   D Loss:{}".format(e, mape_a, trainG_loss, trainD_loss))
    train_end_time = time.time() - start_time
    # save the model 
    # torch.save(modelG.state_dict(),'RTX2080/best_saved_models/' + str(DATASET_NAME) + '.pth')

    # # TESTING FOR MODEL A (train dataset)
    train_inference_start_time = time.time() - start_time
    x_input, truth , predicted = test_tf_enc_dec(test_dl=train_datasetA, model=modelG, scaler=scaler, forecasting_step = seq2seq_dataset_params['target_stride'])
    train_inference_end_time = time.time() - start_time

    # x_input, truth , predicted = test_model(test_dl = train_datasetA, model = modelG, scaler = scaler)
    for i in range(len(truth)):
        truth[i] = truth[i].flatten()
        predicted[i] = predicted[i].flatten()
    truth = np.hstack(truth)
    predicted = np.hstack(predicted)

    train_mape = mape_a = np.round(MAPE(truth, predicted),2)
    train_average_job_count = sum(torch.reshape(torch.FloatTensor(scaler.inverse_transform(_train_data)),(-1,1))) / len(torch.reshape(_train_data,(-1,1)))
    train_rmse = rmse_a = np.round(np.mean(RMSE(truth, predicted)),2) / train_average_job_count
    train_mase = MASE(truth, predicted) 

    # plot_output(path = EXP_FOLDER_PATH, model_name = modelG.model_name, x_input = x_input, truth = truth, predicted = predicted, dataset_params = seq2seq_dataset_params, model_params = "seq2seq_modelG_params", time_interval = time_interval, epochs = epochs, lr = lr, output_for = 'train')
    print("\nTrain Dataset MAPE:{}      RMSE:{}     MASE:{} ".format(mape_a, rmse_a, MASE(truth, predicted)))
    
    train_df = [np.ceil(truth.tolist()), np.ceil(predicted.tolist())]
    train_df = pd.DataFrame(train_df,index=['truth','predicted']).transpose()
    train_df.to_csv('RTX2080/prediction-outputs/'+str(DATASET_NAME)+'/'+str(DATASET_NAME)+str('_train.csv'))

    # # TESTING FOR MODEL A (cross val dataset)
    cv_inference_start_time = time.time() - start_time
    x_input, truth , predicted = test_tf_enc_dec(test_dl=val_datasetA, model=modelG, scaler=scaler, forecasting_step = seq2seq_dataset_params['target_stride'])
    cv_inference_end_time = time.time() - start_time
    # x_input, truth , predicted = test_model(test_dl = val_datasetA, model = modelG, scaler = scaler)

    for i in range(len(truth)):
        truth[i] = truth[i].flatten()
        predicted[i] = predicted[i].flatten()
    truth = np.hstack(truth)
    predicted = np.hstack(predicted)

    cv_mape = mape_a = np.round(MAPE(truth, predicted),2)
    cv_average_job_count = sum(torch.reshape(torch.FloatTensor(scaler.inverse_transform(_cross_val_data)),(-1,1))) / len(torch.reshape(_cross_val_data,(-1,1)))
    cv_rmse = rmse_a = np.round(np.mean(RMSE(truth, predicted)),2) /  cv_average_job_count
    cv_mase = MASE(truth, predicted) 

    # plot_output(path = EXP_FOLDER_PATH, model_name = modelG.model_name, x_input = x_input, truth = truth, predicted = predicted, dataset_params = seq2seq_dataset_params, model_params = "seq2seq_modelG_params", time_interval = time_interval, epochs = epochs, lr = lr, output_for = 'cross_val')
    print("\nCross Val Dataset MAPE:{}      RMSE:{}     MASE:{} ".format(mape_a, rmse_a, MASE(truth, predicted)))

    cv_df = [np.ceil(truth.tolist()), np.ceil(predicted.tolist())]
    cv_df = pd.DataFrame(cv_df,index=['truth','predicted']).transpose()
    cv_df.to_csv('RTX2080/prediction-outputs/'+str(DATASET_NAME)+'/'+str(DATASET_NAME)+str('_crossval.csv'))

    # # TESTING FOR MODEL A (train dataset)
    test_inference_start_time = time.time() - start_time
    x_input, truth , predicted = test_tf_enc_dec(test_dl=test_datasetA, model=modelG, scaler=scaler, forecasting_step = seq2seq_dataset_params['target_stride'])
    test_inference_end_time = time.time() - start_time
    # x_input, truth , predicted = test_model(test_dl = test_datasetA, model = modelG, scaler = scaler)

    for i in range(len(truth)):
        truth[i] = truth[i].flatten()
        predicted[i] = predicted[i].flatten()
    truth = np.hstack(truth)
    predicted = np.hstack(predicted)

    test_mape = mape_a = np.round(MAPE(truth, predicted),2)
    test_average_job_count = sum(torch.reshape(torch.FloatTensor(scaler.inverse_transform(_test_data)),(-1,1))) / len(torch.reshape(_test_data,(-1,1)))
    test_rmse = rmse_a = np.round(np.mean(RMSE(truth, predicted)),2) / test_average_job_count
    test_mase = MASE(truth, predicted) 

    # plot_output(path = EXP_FOLDER_PATH, model_name = modelG.model_name, x_input = x_input, truth = truth, predicted = predicted, dataset_params = seq2seq_dataset_params, model_params = "seq2seq_modelG_params", time_interval = time_interval, epochs = epochs, lr = lr, output_for = 'test')
    print("\nTest Dataset MAPE:{}      RMSE:{}     MASE:{} ".format(mape_a, rmse_a, MASE(truth, predicted)))

    test_df = [np.ceil(truth.tolist()), np.ceil(predicted.tolist())]
    test_df = pd.DataFrame(test_df,index=['truth','predicted']).transpose()
    test_df.to_csv('RTX2080/prediction-outputs/'+str(DATASET_NAME)+'/'+str(DATASET_NAME)+str('_test.csv'))

    del modelG
    del modelD
    del optimizer_G
    del optimizer_D

    training_time   = train_end_time - train_start_time
    train_inference_time = train_inference_end_time - train_inference_start_time
    cv_inference_time = cv_inference_end_time - cv_inference_start_time
    test_inference_time = test_inference_end_time - test_inference_start_time 

    # write data to csv datastructure 
    results = np.append(results, [[str(in_workload_key), str(epochs), str(lr), str(lossG_type), str(seq2seq_dataset_params['window_size']), str(seq2seq_dataset_params['batch_size']), str(seq2seq_modelG_params['dropout']),
                                   str(seq2seq_modelG_params['d_model']), str(seq2seq_modelG_params['nhead']), str(train_mape), str(train_rmse), str(train_mase), str(cv_mape), str(cv_rmse), str(cv_mase),
                                   str(test_mape), str(test_rmse), str(test_mase), str(_gpu_name),
                                   str(training_time), str(train_inference_time), str(cv_inference_time), str(test_inference_time)
                                   ]], axis = 0 )
    
    return results


with open(str(input_workload[in_workload_key])) as csvfile:
    reader = csv.DictReader(csvfile)
    print(reader)
    count = 0
    testNo = 0
    dataNpArray = np.empty([0, 2], int)
    scaleUp = 1
    for row in reader:
        # print(row)
        jobCount = float(row['JobCount'])//float(scaleUp)
        count = int(count) + 1
        testNo = int(testNo) + 1
        dataNpArray = np.append(dataNpArray, [[count, jobCount]], axis=0)

index = [str(i) for i in range(1, len(dataNpArray) + 1)]
data_df = pd.DataFrame(dataNpArray, index=index, columns=['timePeriod', 'jobCount'])

job_arrival_count = data_df[['jobCount']].values.astype('float32')

dataset_name = in_workload_key

# min-max scaling 
scaler = MinMaxScaler(feature_range=(-1,1))

# create test/validation/test 60-15-15 split
split_size = 20 # to create 80-20 split for the training/testing dataset
test_data_size = int(math.ceil(len(job_arrival_count) * split_size / 100))

_train_data1 = job_arrival_count[:-test_data_size]
_test_data = job_arrival_count[-test_data_size:]

cv_data_size = int(math.ceil(len(_train_data1) * 0.25))
_train_data = _train_data1[:-cv_data_size]
_cross_val_data = _train_data1[-cv_data_size:]

test_len = len(_test_data)
print(len(_train_data), len(_cross_val_data), len(_test_data))

_train_data = scaler.fit_transform(_train_data.reshape(-1,1))
_test_data = scaler.fit_transform(_test_data.reshape(-1,1))
_cross_val_data = scaler.fit_transform(_cross_val_data.reshape(-1,1))

# convert data to 1D tensor
_train_data = torch.FloatTensor(_train_data)
_cross_val_data = torch.FloatTensor(_cross_val_data)
_test_data = torch.FloatTensor(_test_data)
    
lookback_set = np.round(np.arange(1,9,1) * 0.1 * test_len)         
print(lookback_set)
experiment_params = [[WINDOW_SIZE, BATCH_SIZE]]

results = np.empty([0,23],str)

for _window_size, _batch_size in experiment_params:
    _results = call_main(_window_size, _batch_size, _train_data, _cross_val_data, _test_data, GPU)#sys.argv[4])            # <----- call to the main func here
    results = np.append(results, _results, axis = 0)

index = [str(i) for i in range(1, len(results) + 1)]
data_df = pd.DataFrame(results, index=index, columns=['Dataset','Epoch','learning_rate','Cost function','Window Size','Batch Size', 'Dropout',
                                                                  'd_model','nhead','Train MAPE','Train RMSE','Train MASE','CV MAPE','CV RMSE','CV MASE','Test MAPE','Test RMSE', 'Test MASE', 'GPU Name',
                                                                  'training-time','train-inference-time','cv-inference-time','test-inference-time'
                                                                  ])
curr_file = pd.read_csv(EXP_FOLDER_PATH + str(DATASET_NAME)+".csv", usecols=['Dataset','Epoch','learning_rate','Cost function','Window Size','Batch Size', 'Dropout',
                                                                  'd_model','nhead','Train MAPE','Train RMSE','Train MASE','CV MAPE','CV RMSE','CV MASE','Test MAPE','Test RMSE', 'Test MASE', 'GPU Name',
                                                                  'training-time','train-inference-time','cv-inference-time','test-inference-time'
                                                                  ])

data_df = data_df.append(curr_file)
data_df.to_csv(EXP_FOLDER_PATH + str(DATASET_NAME)+".csv")
