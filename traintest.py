import time
import sys
import torch
import numpy as np

import metrics as met
import visualize as vis
import create_datasets as cd


# traintest settings
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

show_loss = 1
show_auc = 1
plot_loss = 1
plot_samples = 0
plot_spec = 0
compare_power = 0
save_forecasts = 1


def train(model, train_loader, test_loader, criterion, optimizer, n_epochs, batches_per_epoch, model_name, batch_size):

    print('Start train')
    loss_plot = np.empty(int(batches_per_epoch * n_epochs))
    auc_plot = np.empty(int(batches_per_epoch * n_epochs))
    vloss_plot = np.zeros(int(batches_per_epoch * n_epochs))
    vauc_plot = np.zeros(int(batches_per_epoch * n_epochs))
    yhat_tosave = np.zeros((int(n_epochs), int(batches_per_epoch), batch_size))
    y_tosave = np.zeros((int(n_epochs), int(batches_per_epoch), batch_size))
    yhat_test_tosave = np.array([[]])
    y_test_tosave = np.array([[]])
    t0 = time.time()
    for epoch in range(n_epochs):

        cost = 0
        batch = 0

        for batch_ind, (x, y) in enumerate(train_loader):

            # x, y = x.to(device), y.to(device)
            # Train on batch
            # print('First Batch')
            optimizer.zero_grad()  # clear gradient
            z = model(x)  # make prediciton
            loss = criterion(z, y)  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # update parameters
            cost += loss.item()

            # Save aucs and loss
            plot_index = int(batch + batches_per_epoch * epoch)
            metrics_text = ''
            if show_auc:
                y_np = y.detach().numpy()
                z_np = z.detach().numpy()
                sz_yhat, inter_yhat = met.split_yhat(y_np, z_np)
                a, _, _= met.auc(sz_yhat, inter_yhat)
                auc_plot[plot_index] = a
                metrics_text += 'AUC: %.2g ' % a
            if show_loss:
                loss_plot[plot_index] = loss.item()
                metrics_text += 'loss: %.3g ' % loss.item()
            if save_forecasts:
                y_np = y.detach().numpy()
                yhat_np = z.detach().numpy()
                yhat_tosave[epoch, batch_ind, :y_np.shape[0]] = yhat_np.flatten()
                y_tosave[epoch, batch_ind, :y_np.shape[0]] = y_np.flatten()



            # print
            batch += 1
            t = time.time() - t0
            percent_done = batch / (batches_per_epoch*n_epochs) + epoch/n_epochs
            print('Epoch %d of %d, Batch %d of %d, %0.1f done, %0.2f of %0.2f seconds. ' % (
                epoch + 1, n_epochs, batch, batches_per_epoch, percent_done * 100, t, t / percent_done) + metrics_text)
            # sys.stdout.write('\rBatch %d of %d, %0.1f done, %0.2f of %0.2f seconds. ' % (
            #     batch, batches_per_epoch, percent_done * 100, t, t / percent_done) + metrics_text)

        if plot_loss:
            val_loss = 0
            print('Calculating Test Loss')

            z_np_ = np.array([])
            y_np_ = np.array([])

            for x_, y_ in test_loader:
                z_ = model(x_)

                l = criterion(z_, y_)
                vloss_ind = int(batches_per_epoch*(epoch+1))-1

                sz_, inter_ = met.split_yhat(y_.detach().numpy(), z_.detach().numpy())
                a, _, _ = met.auc(sz_, inter_)

                z_np_ = np.append(z_np, z_.detach().numpy().flatten())
                y_np_ = np.append(y_np, y_.detach().numpy().flatten())


            vloss_plot[vloss_ind] = l.item()
            vauc_plot[vloss_ind] = a
            vis.loss_and_auc(loss_plot, auc_plot, vloss_plot, vauc_plot, model_name, batches_per_epoch, n_epochs)

            if save_forecasts:
                if epoch > 0:
                    yhat_test_tosave = np.append(yhat_test_tosave, np.reshape(z_np_, (1,z_np_.size)), axis=0)
                    y_test_tosave = np.append(y_test_tosave, np.reshape(y_np_, (1,y_np_.size)), axis=0)
                else:
                    yhat_test_tosave = np.append(yhat_test_tosave, np.reshape(z_np_, (1,z_np_.size)), axis=1)
                    y_test_tosave = np.append(y_test_tosave, np.reshape(y_np_, (1,y_np_.size)), axis=1)



        print('--')

    if save_forecasts:
        np.save('/media/projects/daniel_lstm/forecasts_training/' + model_name + '_yhat', yhat_tosave)
        np.save('/media/projects/daniel_lstm/forecasts_training/' + model_name + '_y', y_tosave)
        np.save('/media/projects/daniel_lstm/forecasts_training/' + model_name + '_yhat_t', yhat_test_tosave)
        np.save('/media/projects/daniel_lstm/forecasts_training/' + model_name + '_y_t', y_test_tosave)


    return model


def test(pt, trained_model, validation_loader):
    correct = 0
    batch = 0
    y_tensor = torch.empty(0, dtype=torch.float)
    yhat_tensor = torch.empty(0, dtype=torch.float)
    sample_number = 0

    if compare_power:
        power = np.array([])
        predictions = np.array([])

    for x, y in validation_loader:
        y_tensor = torch.cat((y_tensor, y.type(torch.float)))
        # x, y = x.to(device), y.to(device)
        batch += 1
        sys.stdout.write('\rBatch %d' % batch)

        yhat = trained_model(x)  # was z
        # print('yhat', yhat.data)
        # _, yhat = torch.max(z.data, 1.)
        # yhat = yhat.cpu()
        yhat_tensor = torch.cat((yhat_tensor, yhat.type(torch.float)))

        if plot_samples:
            x_np = x.detach().numpy()
            y_np = y.detach().numpy()
            yhat_np = yhat.detach().numpy()
            print('X shape: ', x_np.shape)
            for i, sample in enumerate(x_np):
                sample_number+=1
                label = y_np[i]
                prediction = yhat_np[i][0]

                # print('Label/Prediction: %d %.2g' % (label, prediction))
                vis.plot_sample(pt, sample, label, prediction, sample_number)
                # cd.spectrogram(x.detach()[i])
        if compare_power:
            x_np = x.detach().numpy()
            yhat_np = yhat.detach().numpy()
            batch_power = np.zeros(x_np.shape[0])
            predictions = np.concatenate((predictions, np.squeeze(yhat_np)))
            for i, sample in enumerate(x_np):
                batch_power[i] = abs(x_np[i]).mean()
            power = np.concatenate((power, batch_power))

    if compare_power: vis.correlation(power, predictions, 'power', 'prediction')


    return y_tensor.detach().numpy(), yhat_tensor.detach().numpy()


def replacement_forward_short(self, x):

    # print('X shape in Short start ', x.detach().numpy().shape)
    print('In replacement')
    batch_size = x.detach().numpy().shape[0]
    sequence_length = 10 * self.lookBack
    num_features = self.num_features
    input_channels = 16
    sample_trim = 239600 * self.lookBack

    self.h0 = torch.randn(1, batch_size, self.rn1)
    self.c0 = torch.randn(1, batch_size, self.rn1)

    with torch.no_grad():
        print('x at start', x.detach().numpy().shape)
        x = x[:, :, :sample_trim]  # trim to reliable number
        print('x after trim', x.detach().numpy().shape)

        x = x.view((batch_size, input_channels, sequence_length, int(sample_trim / sequence_length)))
        x = x.transpose(1, 2)  # (batch_size, seq_length, channels, t)
        x = x.reshape((batch_size * sequence_length, input_channels, int(sample_trim / sequence_length)))
        print('x before transform', x.detach().numpy().shape)

        if self.transform:
            # print(x.size)
            x_ = torch.empty((batch_size * sequence_length, input_channels, 120, 120), dtype=torch.float)

            for sample in range(batch_size * sequence_length):
                x_[sample] = self.transform(x[sample])
            x = x_
        print('x at after transform', x.detach().numpy().shape)

        x = self.min_model(x)  # (batch_size*seq_length, 1)

        print('x after minmodel', x.detach().numpy().shape)

        x = x.view(batch_size, sequence_length, num_features)  # (Batch_size, seq_length)

    # x = x.view((batch_size, sequence_length, num_features))  # (batch_size, seq_length, num_features)
    x, (h1, c1) = self.rnn1(x, (self.h0, self.c0))  # (batch_size, seq_length, hidden_size)
    # x=torch.relu(x)
    x = x.reshape((batch_size * sequence_length, self.rn1))  # (batch_size * seq_length, hidden_size)

    # x = x.view((batch_size*sequence_length, self.rn1))  # (batch_size * seq_length, hidden_size)

    x = self.fc1(x)
    # x = torch.relu(x)
    x = self.bn1(x)

    x = self.fc2(x)

    out = self.sigmoid(x)  # (batch_size * seq_length, hidden_size)
    out = out.reshape((batch_size, sequence_length, 1))  # (batch_size * seq_length, hidden_size)

    out = out[:, -2, :]

    return out


def replacement_forward_medium(self, x):
    batch_size = x.detach().numpy().shape[0]
    sequence_length = self.hrsBack + 1
    num_features=self.num_features
    input_channels=16
    sample_trim = 23960

    self.h0 = torch.randn(1, batch_size, self.rn1)
    self.c0 = torch.randn(1, batch_size, self.rn1)

    with torch.no_grad():

        # input shape: (batch_size, sequence_length, input_channels, sample_trim)
        x = x.reshape((batch_size*sequence_length, input_channels, sample_trim))

        if self.transform:
            x_ = torch.empty((batch_size*sequence_length, input_channels, 120, 120), dtype=torch.float)

            for sample in range(batch_size*sequence_length):
                x_[sample] = self.transform(x[sample])
            x = x_

        x = self.min_model(x)
        x = x.view(batch_size, sequence_length, num_features)

    x, (h1, c1) = self.rnn1(x, (self.h0, self.c0))
    # x = torch.relu(x) # may remove
    x = x.reshape((batch_size*sequence_length, self.rn1))

    x = self.fc1(x)
    x = torch.relu(x)
    x = self.bn1(x)

    x =self.fc2(x)

    out = self.sigmoid(x)
    out = out.reshape((batch_size, sequence_length, 1))

    out = out[:, -2, :]  # out[:, -1, :]

    return out


def replacement_forward_long(self, x):

    batch_size = x.detach().numpy().shape[0]
    sequence_length = self.days_back + 1
    num_features = self.num_features
    input_channels = 16
    sample_trim = 23960

    self.h0 = torch.randn(1, batch_size, self.rn1)
    self.c0 = torch.randn(1, batch_size, self.rn1)

    with torch.no_grad():

        x = x.reshape((batch_size*sequence_length, input_channels, sample_trim))

        if self.transform:
            x_ = torch.empty((batch_size*sequence_length, input_channels, 120, 120), dtype=torch.float)

            for sample in range(batch_size*sequence_length):
                x_[sample] = self.transform(x[sample])
            x = x_

        x = self.min_model(x)
        x = x.view((batch_size, sequence_length, num_features))

    x, (h1, c1) = self.rnn1(x, (self.h0, self.c0))

    x = x.reshape((batch_size*sequence_length, self.rn1))

    x = self.fc1(x)
    x = torch.relu(x)
    x = self.bn1(x)

    x = self.fc2(x)
    out = self.sigmoid(x)
    out = out.reshape((batch_size, sequence_length, 1))

    out = out[:, -2, :]  # out[:, -1, :]

    return out