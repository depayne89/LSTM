import time
import sys
import torch


def train(model, train_loader, criterion, optimizer, n_epochs, batches_per_epoch):
    for epoch in range(n_epochs):
        t0 = time.time()
        cost = 0
        batch = 0
        for x, y in train_loader:
            batch += 1
            t = time.time() - t0
            percent_done = batch / batches_per_epoch
            sys.stdout.write('\rBatch %d of %d, %0.1f done, %0.2f of %0.2f seconds' % (
            batch, batches_per_epoch, percent_done * 100, t, t / percent_done))

            optimizer.zero_grad()  # clear gradient
            z = model(x)  # make prediciton
            loss = criterion(z, y)  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # update parameters
            cost += loss.item()

    return model


def test(trained_model, validation_loader):
    correct = 0
    batch = 0
    y_tensor = torch.empty(0, dtype=torch.float)
    yhat_tensor = torch.empty(0, dtype=torch.float)
    for x, y in validation_loader:
        batch += 1
        sys.stdout.write('\rBatch %d' % batch)

        yhat = trained_model(x)  # was z
        # print('yhat', yhat.data)
        # _, yhat = torch.max(z.data, 1.)
        y_tensor = torch.cat((y_tensor, y.type(torch.float)))
        yhat_tensor = torch.cat((yhat_tensor, yhat.type(torch.float)))
    return y_tensor.detach().numpy(), yhat_tensor.detach().numpy()