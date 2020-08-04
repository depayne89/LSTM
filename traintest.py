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


def train(model, train_loader, test_loader, criterion, optimizer, n_epochs, batches_per_epoch, model_name):

    loss_plot = np.empty(int(batches_per_epoch * n_epochs))
    auc_plot = np.empty(int(batches_per_epoch * n_epochs))
    vloss_plot = np.zeros(int(batches_per_epoch * n_epochs))
    vauc_plot = np.zeros(int(batches_per_epoch * n_epochs))
    t0 = time.time()
    for epoch in range(n_epochs):

        cost = 0
        batch = 0

        for x, y in train_loader:

            # x, y = x.to(device), y.to(device)
            # Train on batch
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
            for x_, y_ in test_loader:
                z_ = model(x_)

                l = criterion(z_, y_)
                vloss_ind = int(batches_per_epoch*(epoch+1))-1

                sz_, inter_ = met.split_yhat(y_.detach().numpy(), z_.detach().numpy())
                a, _, _ = met.auc(sz_, inter_)

            vloss_plot[vloss_ind] = l.item()
            vauc_plot[vloss_ind] = a
            vis.loss_and_auc(loss_plot, auc_plot, vloss_plot, vauc_plot, model_name)




        print('--')
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