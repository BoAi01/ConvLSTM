import csv
import os

import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """
    Log values to file
    """

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values, f'{col} not in {values}'
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def sanity_check(model, input, labels, criterion, optimizer, pass_pred, nepoch=50):
    input, labels, model = input.cuda(), labels.cuda(), model.cuda()
    loss_values = []
    for i in range(nepoch):
        optimizer.zero_grad()
        out = model(input)
        los = criterion(out, labels)
        los.backward()
        optimizer.step()
        loss_values.append(los.item())
    print(f'sanity check losses: {loss_values}')
    return pass_pred(loss_values[-1])


def save_model(ckpt_dir, cp_name, model):
    """
    Create directory /Checkpoint under exp_data_path and save model as cp_name
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    saving_model_path = os.path.join(ckpt_dir, cp_name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # convert to non-parallel form
    torch.save(model.state_dict(), saving_model_path)
    print(f'Model saved: {saving_model_path}')


def load_model_dic(model, ckpt_path, verbose=True):
    """
    Load weights to model and take care of weight parallelism
    """
    assert os.path.exists(ckpt_path), f"trained model {ckpt_path} does not exist"

    try:
        model.load_state_dict(torch.load(ckpt_path))
    except:
        state_dict = torch.load(ckpt_path)
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict)
    if verbose:
        print(f'Model loaded: {ckpt_path}')

    return model


def read_log(data_path):
    with open(data_path) as f:
        lines = f.readlines()
        epoch = [float(line.split()[0]) for line in lines[1:]]
        loss = [float(line.split()[1]) for line in lines[1:]]
        if len(lines[0].split()) > 2:
            lr = [float(line.split()[2]) for line in lines[1:]]
            return epoch, loss, lr
    return epoch, loss
