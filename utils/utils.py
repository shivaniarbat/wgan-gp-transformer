from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
# from loss.dilate_loss import DILATE_loss
from sklearn.metrics import mean_squared_error
import math

def minmax(data):
    r"""Returns transformed data and scaler"""
    scaler = MinMaxScaler(feature_range=(-1,1))
    data_transformed= scaler.fit_transform(data.reshape(-1,1))

    return scaler, data_transformed

def train(model, criterion, optimizer, train_dl):
    model.train()
    train_loss = 0
    n= 0 
    for x,y,mask in train_dl:
        optimizer.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')
        output = model(x).float()

        # loss = criterion(output.squeeze(-1),y.squeeze(-1))
        loss = criterion(output,y.squeeze(-1))

        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * x.shape[0])
        n += x.shape[0]

        del loss
        del output

    return train_loss/n  

def train_encdec(model, criterion, optimizer, train_dl, scaler):
    model.train()
    train_loss = 0

    x_input = []
    truth = []
    predicted = []

    n= 0 
    for x,y,tgt_mask in train_dl:
        optimizer.zero_grad()
        x = x.to('cuda')
        y = y.unsqueeze(-1).to('cuda')
        tgt_mask = tgt_mask.to('cuda')
        output = model(x,y,tgt_mask[0]).float()                # def forward(self, src, tgt, tgt_mask):

        # loss = criterion(output.squeeze(-1),y.squeeze(-1))
        loss = criterion(output,y.squeeze(-1))

        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * x.shape[0])
        n += x.shape[0]

        x = x.to('cpu')
        y = y.to('cpu')
        output = output.to('cpu')

        x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).detach().numpy()),(x.shape[1],1))))
        truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).detach().numpy()),(y.shape[1],1))))
        predicted.append(scaler.inverse_transform(np.reshape(np.array(output[0].detach().view(-1).numpy()),(output.shape[1],1))))


    return train_loss/n, x_input, truth, predicted 

def eval(model, criterion, optimizer, eval_dl):
    model.eval()
    eval_loss = 0
    n= 0 
    for step, (x,y,mask) in enumerate(eval_dl):
        optimizer.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')
        output = model(x).float()
        loss = criterion(output.squeeze(-1),y.squeeze(-1))

        eval_loss += (loss.item() * x.shape[0])
        n += x.shape[0]
        
        del loss
        del output

    return eval_loss/n

def test_model(test_dl, model, scaler):
    """Return input, truth and prediction for input model and data"""
    x_input = []
    truth = []
    predicted = []

    with torch.no_grad():
        model.eval()
        step = 0
    
        for x,y,mask in test_dl:
            x = x.to('cuda')
            y = y.to('cuda')
            output = model(x).float()

            x = x.to('cpu')
            y = y.to('cpu')
            output = output.to('cpu')
            
            x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
            truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
            predicted.append(scaler.inverse_transform(np.reshape(np.array(output[0].view(-1).numpy()),(output.shape[1],1))))
            # scaler for better visualization
    return x_input, truth, predicted

def test_enc_dec_model_seq_at_a_time(test_dl, model, scaler, output_sequence_length):         
    """Return input, truth and prediction for input model and data"""
    x_input = []
    truth = []
    predicted = []

    with torch.no_grad():
        model.eval()
        step = 0
    
        for x,y,mask in test_dl:
            x = x.to('cuda')
            y = y.unsqueeze(-1).to('cuda')
            tgt_mask = mask.to('cuda')
            memory = model.encoder(model.positional_encoding(x))

            # outputs = torch.zeros_like(y)
            outputs = torch.clone(x)

            # out = model.out(model.decoder(model.positional_encoding(outputs),memory, tgt_mask))

            # print(out.shape)
            for i in range(1,output_sequence_length+1,1):
                mask = (torch.triu(torch.ones(1, 1)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                tgt_mask = mask.to('cuda')

                if i == 1:
                    dec_in = outputs[:i].unsqueeze(-1)
                else:
                    dec_in = outputs[i : i+1]
                dec_in = dec_in.unsqueeze(-1).to('cuda')

                out = model.out(model.decoder(model.positional_encoding(dec_in),memory, tgt_mask))

                outputs[i] = out.view(-1).squeeze(-1)
            # output = model(x,y,tgt_mask[0]).float()  
            # output = output.to('cpu')
            x = x.to('cpu')
            y = y.to('cpu')
            out = out.to('cpu')

            x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
            truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
            predicted.append(scaler.inverse_transform(np.reshape(np.array(out[0].view(-1).numpy()),(out.shape[1],1))))
                # scaler for better visualization
    return x_input, truth, predicted

def test_dec_one_ts_a_time(test_dl, model, scaler, output_sequence_length):         
    """Return input, truth and prediction for input model and data"""
    """
    This function is especially for transformer encoder decoder architecture. This method tests the transformer encoder decoder model 
    """
    x_input = []
    truth = []
    predicted = []

    with torch.no_grad():
        model.eval()
        step = 0
    
        for x,y,mask in test_dl:
            x = x.to('cuda')
            y = y.unsqueeze(-1).to('cuda')
            tgt_mask = mask.to('cuda')

            memory = model.encoder(model.positional_encoding(x))
            # outputs = torch.zeros_like(y)
            outputs = y.clone()

            # print(out.shape)
            for i in range(1,output_sequence_length+1,1):
                mask = (torch.triu(torch.ones(i, i)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                tgt_mask = mask.to('cuda')

                dec_input = outputs[:i].to('cuda')#.unsqueeze(-1)
                out = model.out(model.decoder(model.positional_encoding(dec_input),memory, tgt_mask))
                
                outputs[:i:] = out
                
            x = x.to('cpu')
            y = y.to('cpu')
            output = outputs.to('cpu')

            x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
            truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
            predicted.append(scaler.inverse_transform(np.reshape(np.array(output[0].view(-1).numpy()),(output.shape[1],1))))
                # scaler for better visualization
    return x_input, truth, predicted

def plot_loss(path= None,model_name= None,train_epoch_loss= None, eval_epoch_loss = None, model_params= None, lr= None, epochs= None, time_interval= None):
    plt.figure(figsize=(15,5))
    plt.plot(train_epoch_loss)
    if eval_epoch_loss is not None:
        plt.plot(eval_epoch_loss)
        plt.legend(["training loss", "validation loss"])
    else:
        plt.legend(["training loss"])
    plt.xlabel("Epochs",fontsize=10)
    plt.ylabel("MSE Loss",fontsize=10)
    plt.title(str(model_name)+" "+str(model_params) + "{learning rate:" + str(lr) +" epochs:" + str(epochs) + "}" , fontsize=10)
    plt.savefig(path + str(model_name)+'_MSE_loss_'+ str(time_interval) + ' mins ts.png')
    plt.close()

def plot_output(path,model_name,x_input, truth, predicted, dataset_params, model_params, time_interval, epochs, lr, output_for = '_test_output_', figtext = None):
    # plt.figure(figsize=(30,10))
    # plt.plot(torch.Tensor(x_input).view(-1),'b',linewidth=1)
    plt.plot(np.arange(dataset_params['window_size'],torch.Tensor(truth).view(-1).numpy().shape[0] + int(dataset_params['window_size']),1),torch.Tensor(truth).view(-1) ,'g',linewidth=1)
    plt.plot(np.arange(dataset_params['window_size'],torch.Tensor(predicted).view(-1).numpy().shape[0] + int(dataset_params['window_size']),1),torch.Tensor(predicted).view(-1),'r',linewidth=1)
    plt.grid(axis='x')
    # plt.legend(["${input}$", "${predicted}$", "${truth}$"])
    plt.legend(["${truth}$", "${predicted}$"])
    plt.ylabel("job arrival count {scaled}")#, fontsize=20)
    plt.xlabel("time steps {sample rate "+ str(time_interval) +" min interval)")#, fontsize=20)
    if output_for == '_test_output_':
        plt.suptitle(str (model_name) + " { Test output Google Cluster }")#, fontsize=20)
    else:
        plt.suptitle(str(model_name) + " { Train output Google Cluster }")#, fontsize=20)
    if figtext != None:
        plt.figtext(.835,.8, str(figtext))
    plt.title(str(model_name)+" "+str(model_params) + "{learning rate:" + str(lr) +" epochs:" + str(epochs) + "}")#, fontsize=10)
    plt.savefig(path+str(model_name)+ str(output_for) + str(time_interval) + 'mins_ts.png')
    plt.close()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_square_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # averageTest = sum(y_true) / len(y_true)
    score = np.sqrt(np.mean((y_pred - y_true)**2)) #/ averageTest
    # return np.sqrt(np.mean((y_pred - y_true)**2))
    return score

def mean_absolute_scaled_error(y_true, y_pred):
    ''' (y_pred - target).abs() / all_targets[:, :-1] - all_targets[:, 1:]).mean(1)
    MASE is the mean absolute error of the forecast values, divided by the mean absolute error of the in-sample one-step naive forecast
    '''
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    n = y_true.shape[0] 
    d = np.abs(  np.diff(y_true, axis=-1) ).sum()/(n-1)       # Not applicable : differencing is done at axis = 1 beacuse we have a a 3D array [window size, batch_size, features]
    errors = np.abs(y_true - y_pred)
    return errors.mean()/d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict(model, input_sequence, forecasting_step, input_window):
    forecast_output = input_sequence[-1].numpy()
    with torch.no_grad():
        for i in range(forecasting_step):
            data = torch.reshape(torch.Tensor(forecast_output[-input_window:]), (1,input_window,1)).to('cuda')
            output = model(data)
            output = output.cpu()
            # input_sequence = torch.cat((input_sequence, output[-1:].unsqueeze(-1)))
            forecast_output = np.append(forecast_output, output[-1].numpy())
        # input_sequence = input_sequence.cpu().view(-1)
    # print(forecast_output)
    return forecast_output

def train_tf_encdec(model, criterion, optimizer, train_dl, scaler, forecasting_step, dilate_loss_params = None):
    """
        This function is a to train the model by feeding last ts of x as first input to decoder to predict first ts. 
        Then the output of decoder is fed as input to decoder to predict the next ts and so on.
    """
    model.train()
    train_loss = 0

    shape_l = 0
    temporal_l = 0

    x_input = []
    truth = []
    predicted = []

    n = 0

    # criterion = torch.nn.MSELoss() 
    # mae_criterion = torch.nn.L1Loss()

    # initialize losses
    mse_loss, shape_loss, temporal_loss, mae_loss = torch.tensor(0),torch.tensor(0),torch.tensor(0), torch.tensor(0)
    FIRST = False 
    for x, y, tgt_mask in train_dl:
        optimizer.zero_grad()
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

        # if loss == 'mse':
        #     mse_loss = criterion(outputs,y)
        #     loss = mse_loss
        # elif loss == 'dilate_loss':
        #     loss, shape_loss, temporal_loss = dilate_loss(outputs, y, dilate_loss_params['alpha'], dilate_loss_params['gamma'], device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # elif loss == 'mae':
        #     mae_loss = mae_criterion(outputs, y)
        #     loss = mae_loss
        #     loss.backward(retain_graph = True)
        #     FIRST = True

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * x.shape[0])
        n += x.shape[0]
        
        x = x.to('cpu') 
        y = y.to('cpu')
        outputs = outputs.detach().to('cpu')
        
        x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
        truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
        predicted.append(scaler.inverse_transform(np.reshape(np.array(outputs[0].view(-1).numpy()),(outputs.shape[1],1))))
           
        shape_l = (shape_loss.item() * x.shape[0])
        temporal_l = (temporal_loss.item() * x.shape[0])

    return train_loss/n, x_input, truth, predicted, shape_l, temporal_l

def test_tf_enc_dec(test_dl, model, scaler, forecasting_step):
    """Return input, truth and prediction for input model and data"""
    x_input = []
    truth = []
    predicted = []

    with torch.no_grad():
        model.eval()
        
        for x, y, tgt_mask in test_dl:
            x = x.to('cuda')
            y = y.to('cuda')

            # encode x
            enc_out = model.encoder(model.positional_encoding(x)).to('cuda')

            out = torch.clone(x[-1:])   # last bit from the input
            outputs = torch.clone(x[-forecasting_step:])
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
                # print(outputs[step - 1:step:].shape, out.shape)
                outputs[step - 1:step:] = out 

                y = y.unsqueeze(-1)
                x = x.to('cpu') 
                y = y.to('cpu')
                outputs = outputs.detach().to('cpu')
                
            x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
            truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
            predicted.append(scaler.inverse_transform(np.reshape(np.array(outputs[0].view(-1).numpy()),(outputs.shape[1],1))))
           
    return x_input, truth, predicted

def train_tf_enc_dec_gan_XX(model, discriminator,
            optimizer_G,
            optimizer_D,
            criterion,
            adverserial_loss,
            lambda_,
            train_dl,
            epochs,
            modelG_params,
            modelD_params,
            scaler,
            EXP_FOLDER_PATH,
            time_interval,
            lr,
            dataset_params,
            forecasting_step=2):
                
    # THIS TRAINING FUNCTION UPDATES GRADIENT FOR GENERATOR & DISCRIMINATOR FOR EVERY BATCH
    
    ''' Train the generator and discriminator AST similar algorithm with non-sparse attention. 
        Adversarial Sparse Transformer for Time Series Forecasting (http://file.mrdqg.com/NeurIPS-2020-adversarial-sparse-transformer-for-time-series-forecasting-Paper.pdf)

    Args:
        model : transformer encoder decoder network
        discriminator : the discriminator model 
        optimizerG : optimizer for generator network in this case transformer encoder-deocoder model
        optimizerD : optimizer for discriminator network 
        epoch : current epoch
        modelG_params : generator model parameters {hyperparams}
        modelD_params : discriminator model parameters {hyperparams}
    '''
    with open(EXP_FOLDER_PATH+'/training_graphs/training_terminal_output_'+ str(time_interval) +'mins_sr.txt','w') as fp:
        fp.write("Generator Model\n")
        fp.write(str(model.parameters))
        fp.write("\n\nDiscriminator Model\n")
        fp.write(str(discriminator.parameters))
        fp.write("\n------------------------\n")
        fp.write(str(model.parameters))
        for e, epoch in enumerate(range(epochs)):
            model.train()

            trainG_loss = 0
            trainD_loss = 0
            nG = 0 

            x_input = []
            truth = []
            predicted = []

            # discriminator linear encoding 
            linear_encoding = nn.Linear(1,modelD_params['d_model'])

            for x, y, tgt_mask in train_dl:
                x = x.to('cuda')
                y = y.to('cuda')

                # Adversarial ground truths
                valid = torch.autograd.Variable(torch.cuda.FloatTensor(2,64, 1).fill_(1.0), requires_grad=False) # update the hardcoded tensor size
                fake = torch.autograd.Variable(torch.cuda.FloatTensor(2,64, 1).fill_(0.0), requires_grad=False)  # update the hardcoded tensor size

                # >>>> generator processing starts here
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
                # <<<<< generator processing ends here 

                y = y.unsqueeze(-1)

                # ---------------------------------------------
                # Training Transformer Encoder-Decoder Network 
                # ---------------------------------------------
                optimizer_G.zero_grad()

                loss = criterion(outputs, y) + lambda_ * adverserial_loss(discriminator(outputs), valid)
                loss.backward(retain_graph = True)
                optimizer_G.step()

                trainG_loss += (loss.item() * x.shape[0])    # to take mean of total loss
                nG += x.shape[0]

                # ---------------------------------------------
                # Training the Discriminator  
                # ---------------------------------------------
                optimizer_D.zero_grad()
                real_loss = adverserial_loss(discriminator(y), valid)
                fake_loss = adverserial_loss(discriminator(outputs), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward(retain_graph = True)
                optimizer_D.step()

                trainD_loss += (d_loss.item() * x.shape[0])

                # return generator output 
                x = x.to('cpu') 
                y = y.to('cpu')
                outputs = outputs.detach().to('cpu')
                
                x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
                truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
                predicted.append(scaler.inverse_transform(np.reshape(np.array(outputs[0].view(-1).numpy()),(outputs.shape[1],1))))

            if e % 5 == 0 or e == (epochs - 1):
                mape_a = np.round(mean_absolute_percentage_error(truth, predicted),2)
                print("Epoch: {} => MAPE:{} ; loss_G: {} ; loss_D: {}".format(e,mape_a, trainG_loss/nG, trainD_loss/nG))
                to_write = "\nEpoch: {} => MAPE:{} ; loss_G: {} ; loss_D: {}".format(e,mape_a, trainG_loss/nG, trainD_loss/nG)
                fp.write(to_write)
                # plot the graph 
                if e % 100 == 0 or e == (epochs - 1):
                    plot_output(path=EXP_FOLDER_PATH +'training_graphs/'+str(time_interval)+'_mins/epoch_'+str(epoch),model_name=str('MAPE_') + str(mape_a) +' ' + str(model.model_name),x_input=x_input, truth=truth, predicted=predicted, 
                    dataset_params=dataset_params, model_params=modelG_params, time_interval=time_interval, epochs=epochs, lr=lr, output_for='train_output')
        fp.write("\ntraining end here. Model saved to >>>>>> {}".format(EXP_FOLDER_PATH))
        fp.write("\n------------------------\n")
        fp.close # close file which captures training terminal output 
    # save model to make predictions 
    torch.save(model.state_dict(), EXP_FOLDER_PATH + str("saved_model/"+ str(time_interval) +"_mins/"+str(modelG_params['num_of_enc_layers'])+'_layer_'+"seq_2_seq_generator_tran_enc_dec_model_"+str(time_interval)+"mins_sr"))

def train_tf_enc_dec_gan(model, discriminator,
            optimizer_G,
            optimizer_D,
            criterion,
            adverserial_loss,
            lambda_,
            train_dl,
            epochs,
            modelG_params,
            modelD_params,
            scaler,
            EXP_FOLDER_PATH,
            time_interval,
            lr,
            dataset_params,
            forecasting_step=1):
                
    # THIS TRAINING FUNCTION UPDATES GRADIENT FOR GENERATOR for one epoch then updates gradient for Discriminator 
    
    ''' Train the generator and discriminator AST similar algorithm with non-sparse attention. 
        Adversarial Sparse Transformer for Time Series Forecasting (http://file.mrdqg.com/NeurIPS-2020-adversarial-sparse-transformer-for-time-series-forecasting-Paper.pdf)

    Args:
        model : transformer encoder decoder network
        discriminator : the discriminator model 
        optimizerG : optimizer for generator network in this case transformer encoder-deocoder model
        optimizerD : optimizer for discriminator network 
        epoch : current epoch
        modelG_params : generator model parameters {hyperparams}
        modelD_params : discriminator model parameters {hyperparams}
    '''
    with open(EXP_FOLDER_PATH+'training_terminal_output_'+ str(time_interval) +'mins_sr.txt','w') as fp:
        # fp.write("Generator Model\n")
        # fp.write(str(model.parameters))
        # fp.write("\n\nDiscriminator Model\n")
        # fp.write(str(discriminator.parameters))
        print("EPOCHS: {}   WINDOW SIZE: {}     BATCH_SIZE: {}". format(str(1000), dataset_params['window_size'], dataset_params['batch_size']))
        to_write = "EPOCHS: {}   WINDOW SIZE: {}     BATCH_SIZE: {}". format(str(1000), dataset_params['window_size'], dataset_params['batch_size'])
        fp.write(to_write)
        fp.write("\n------------------------\n")
        # fp.write(str(model.parameters))
        for e, epoch in enumerate(range(epochs)):
            model.train()

            trainG_loss = 0
            trainD_loss = 0
            nG = 0 
            nD = 0

            x_input = []
            truth = []
            predicted = []

            # discriminator linear encoding 
            # linear_encoding = nn.Linear(1,modelD_params['d_model'])

            for x, y, tgt_mask in train_dl:
                x = x.to('cuda')
                y = y.to('cuda')

                # Adversarial ground truths
                valid = torch.autograd.Variable(torch.cuda.FloatTensor(dataset_params['batch_size'], 1).fill_(1.0), requires_grad=False) # update the hardcoded tensor size
                fake = torch.autograd.Variable(torch.cuda.FloatTensor(dataset_params['batch_size'], 1).fill_(0.0), requires_grad=False)  # update the hardcoded tensor size

                # >>>> generator processing starts here
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
                # <<<<< generator processing ends here 

                y = y.unsqueeze(-1)

                # ---------------------------------------------
                # Training Transformer Encoder-Decoder Network 
                # ---------------------------------------------
                optimizer_G.zero_grad()
                loss = criterion(outputs, y) + lambda_ * adverserial_loss(discriminator(outputs), valid)
                loss.backward(retain_graph = True)
                optimizer_G.step()

                trainG_loss += (loss.item() * x.shape[0])    # to take mean of total loss
                nG += x.shape[0]

            for x, y, tgt_mask in train_dl:
                x = x.to('cuda')
                y = y.to('cuda')

                # >>>> generator processing starts here
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
                # <<<<< generator processing ends here 

                y = y.unsqueeze(-1)

                # Adversarial ground truths
                valid = torch.autograd.Variable(torch.cuda.FloatTensor(forecasting_step,dataset_params['batch_size'], 1).fill_(1.0), requires_grad=False) # update the hardcoded tensor size
                fake = torch.autograd.Variable(torch.cuda.FloatTensor(forecasting_step,dataset_params['batch_size'], 1).fill_(0.0), requires_grad=False)  # update the hardcoded tensor size

                # ---------------------------------------------
                # Training the Discriminator  
                # ---------------------------------------------
                optimizer_D.zero_grad()
                real_loss = adverserial_loss(discriminator(y), valid)
                fake_loss = adverserial_loss(discriminator(outputs), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward(retain_graph = True)
                optimizer_D.step()

                trainD_loss += (d_loss.item() * x.shape[0])
                nD += x.shape[0]

                # return generator output 
                x = x.to('cpu') 
                y = y.to('cpu')
                outputs = outputs.detach().to('cpu')
                
                x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
                truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
                predicted.append(scaler.inverse_transform(np.reshape(np.array(outputs[0].view(-1).numpy()),(outputs.shape[1],1))))

            if e % 5 == 0 or e == (epochs - 1):
                mape_a = np.round(mean_absolute_percentage_error(truth, predicted),2)
                print("Epoch: {} => MAPE:{} ; loss_G: {} ; loss_D: {}".format(e,mape_a, trainG_loss/nG, trainD_loss/nD))
                to_write = "\nEpoch: {} => MAPE:{} ; loss_G: {} ; loss_D: {}".format(e,mape_a, trainG_loss/nG, trainD_loss/nD)
                fp.write(to_write)
                # plot the graph 
                # if e % 100 == 0 or e == (epochs - 1):
                #     plot_output(path=EXP_FOLDER_PATH +'training_graphs/'+str(time_interval)+'_mins/epoch_'+str(epoch),model_name=str('MAPE_') + str(mape_a) +' ' + str(model.model_name),x_input=x_input, truth=truth, predicted=predicted, 
                #     dataset_params=dataset_params, model_params=modelG_params, time_interval=time_interval, epochs=epochs, lr=lr, output_for='train_output')
        fp.write("\ntraining end here. Model saved to >>>>>> {}".format(EXP_FOLDER_PATH))
        fp.write("\n------------------------\n")
        fp.close # close file which captures training terminal output 
    # save model to make predictions 
    # torch.save(model.state_dict(), EXP_FOLDER_PATH + str("saved_model/"+ str(time_interval) +"_mins/"+str(modelG_params['num_of_enc_layers'])+'_layer_'+"seq_2_seq_generator_tran_enc_dec_model_"+str(time_interval)+"mins_sr"))
    # torch.save(model.state_dict(), EXP_FOLDER_PATH + str("saved_model/"+ "batch_size_" + str(batch_size) + "_lookback_"+str(look_back) + ".pth"))

def train_and_evaluate_tf_enc_dec_gan(model, discriminator,
            optimizer_G,
            optimizer_D,
            criterion,
            adverserial_loss,
            lambda_,
            train_dl,
            test_dl,
            epochs,
            modelG_params,
            modelD_params,
            scaler,
            EXP_FOLDER_PATH,
            time_interval,
            lr,
            dataset_params,
            forecasting_step=1):
    
    train_tf_enc_dec_gan(model, discriminator, optimizer_G, optimizer_D,
            criterion, adverserial_loss,
            lambda_, train_dl, epochs,
            modelG_params, modelD_params,
            scaler, EXP_FOLDER_PATH, time_interval,
            lr, dataset_params, forecasting_step)
    # # Test on TRAIN DATASET on train_dl

    # # TESTING FOR MODEL A
    x_input, truth , predicted = test_tf_enc_dec(test_dl=train_dl, model=model, scaler=scaler, forecasting_step = dataset_params['target_stride'])

    mape_a = np.round(mean_absolute_percentage_error(truth, predicted),2)
    print("Train Dataset MAPE:", mape_a)

    # # plot testing output 
    # plot_output(path=EXP_FOLDER_PATH +'testing_graphs/'+str(time_interval)+'_mins/'+str(layers_num)+'_layer_s_',model_name=str('MAPE_') + str(mape_a) +' ' + str(model.model_name),x_input=x_input, truth=truth, predicted=predicted, 
    #     dataset_params=dataset_params, model_params=modelG_params, time_interval=time_interval, epochs=epochs, lr=lr, output_for='_train_output_')

    # # PERFORM TESTING on test_dl

    # # TESTING FOR MODEL A
    x_input, truth , predicted = test_tf_enc_dec(test_dl=test_dl, model=model, scaler=scaler, forecasting_step = dataset_params['target_stride'])

    mape_a = np.round(mean_absolute_percentage_error(truth, predicted),2)
    print("Test Dataset MAPE:", mape_a)

    # # plot testing output 
    # plot_output(path=EXP_FOLDER_PATH +'testing_graphs/'+str(time_interval)+'_mins/'+str(layers_num)+'_layer_s_',model_name=str('MAPE_') + str(mape_a) +' ' + str(model.model_name),x_input=x_input, truth=truth, predicted=predicted, 
    #     dataset_params=dataset_params, model_params=modelG_params, time_interval=time_interval, epochs=epochs, lr=lr, output_for='_test_output_')