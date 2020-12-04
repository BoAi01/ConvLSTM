#!/usr/bin/env python
# coding: utf-8

import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import RGBDataset_test, RGBDataset
from nets import Net04

root = '/scratch/e0376958'
ckp_root = '/home/svu/e0376958/ConvLSTM/ckp'
NUM_FRAMES = 6
base_batch_size, base_learng_rate = 16, 1e-3
num_workers = 8
start_epoch, end_epoch = 0, 200
checkpoint = list(range(start_epoch, end_epoch, 10))
scale_factor = 1
torch.manual_seed(0)

model = nn.DataParallel(Net04()).cuda()

root_dir = os.path.join(root, 'imgs')
root_dir1 = os.path.join(root_dir, 'img1')
root_dir2 = os.path.join(root_dir, 'img2')
root_dir3 = os.path.join(root_dir, 'img3')

rgb_train = RGBDataset(NUM_FRAMES, root_dir1, root_dir2, root_dir3)  # frame
rgb_test = RGBDataset_test(NUM_FRAMES, root_dir1, root_dir2, root_dir3)

dataloader_train = DataLoader(rgb_train, batch_size=int(base_batch_size * scale_factor), shuffle=False,
                              num_workers=num_workers, pin_memory=True)
dataloader_test = DataLoader(rgb_test, batch_size=int(base_batch_size * scale_factor),
                             shuffle=False, num_workers=num_workers, pin_memory=True)

print(f'num_gpus: {torch.cuda.device_count()}, bs = {int(base_batch_size * scale_factor)}')

# hardware-side settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# sanity check
model = Net04().cuda()
p = torch.randn(1, 6, 3, 112, 336).cuda()
q = torch.randn(5, 1, 2).cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)
loss_values = []
for i in range(50):
    optimizer.zero_grad()
    out = model(p)
    los = criterion(out, q)
    los.backward()
    optimizer.step()
    loss_values.append(los.item())
if loss_values[-1] < loss_values[0] / 10:
    print(f'sanity check passed')
else:
    print(f'sanity check not passed')
del model
print(f'losses: {loss_values}')

# create your optimizer
model = nn.DataParallel(Net04()).cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=base_learng_rate * scale_factor)
loss_values, test_values, epochs = [], [], []

for epoch in range(start_epoch, end_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    running_test_loss = 0.0
    start_time = time.time()
    for i, data in enumerate(dataloader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        X_t, y_t = data['X_train'].cuda(), data['y_train'].cuda()
        optimizer.zero_grad()
        for j in range(len(X_t)):
            outputs = model(X_t[j].reshape(1, NUM_FRAMES, 3, 112, 336))
            if j == 0:
                loss = criterion(outputs, y_t[j].reshape(-1, 1, 2))
            else:
                loss += criterion(outputs, y_t[j].reshape(-1, 1, 2))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_values.append(running_loss)

    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            X_test, y_test = data['X_test'].cuda(), data['y_test'].cuda()
            for j in range(len(X_test)):
                out_test = model(X_test[j].reshape(1, NUM_FRAMES, 3, 112, 336))
                if j == 0:
                    test_loss = criterion(out_test, y_test[j].reshape(-1, 1, 2))
                else:
                    test_loss += criterion(out_test, y_test[j].reshape(-1, 1, 2))
            running_test_loss += test_loss.item()
        test_values.append(running_test_loss)

    epochs.append(epoch)
    end_time = time.time()
    if epoch % 10 == 9:
        torch.save(model.module.state_dict(), os.path.join(ckp_root, f'model_303_fac{scale_factor}_e{epoch}'))
    print('[epoch%d] running_loss: %.3f, test_loss:%.3f, runtime = %.3f' % (
        epoch + 1, running_loss, running_test_loss, (end_time - start_time) / 60))

print('Finished Training')

torch.save(model.module.state_dict(), os.path.join(ckp_root, f'final_model_303_fac{scale_factor}'))

# plot curve
matplotlib.use('Agg')  # use noninteractive backend
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss_values, label='train loss', color='C0')
plt.plot(epochs, test_values, label='test loss', color='C2')
plt.savefig(f'curve_lr{base_learng_rate}.jpg')
