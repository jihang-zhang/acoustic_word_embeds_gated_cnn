import numpy as np
import time
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
from torch.utils.data import DataLoader, TensorDataset

from data import make_loader, load_swbd_labelled
from average_precision import average_precision
from model import GatedCNN

# Configurations
seed = 37
log_interval = 50
MAX_EPOCHS = 40
min_count = 3
batch_size = 32
SAVE_PATH = "../check_points_GatedCNN"
data_dir = "../data"

# Set the random seed manually for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    # Select cuda device
    cuda0 = torch.device('cuda:0')

rng = np.random.RandomState(seed)
random.seed(seed)

def data_loader():
    datasets, word_to_i_map = load_swbd_labelled(rng, data_dir, min_count)
    ntokens = len(word_to_i_map)

    # Loading train set
    train_x, train_y = datasets[0]
    train_x = torch.cuda.FloatTensor(train_x) if torch.cuda.is_available() else torch.FloatTensor(train_x)
    train_y = torch.cuda.LongTensor(train_y) if torch.cuda.is_available() else torch.FloatTensor(train_y)
    train = TensorDataset(train_x, train_y)   # comment out if using get_batch2
    train_ldr = make_loader(train, batch_size)

    # Loading dev and test data
    datasets, _ = load_swbd_labelled(rng, data_dir, 1)
    dev_x, dev_y = datasets[1]
    test_x, test_y = datasets[2]

    dev = TensorDataset(torch.cuda.FloatTensor(dev_x), torch.cuda.LongTensor(dev_y)) if torch.cuda.is_available() else TensorDataset(torch.FloatTensor(dev_x), torch.LongTensor(dev_y))
    test = TensorDataset(torch.cuda.FloatTensor(test_x), torch.cuda.LongTensor(test_y)) if torch.cuda.is_available() else TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
    dev_ldr = make_loader(dev, batch_size)
    test_ldr = make_loader(test, batch_size)

    return ntokens, train_ldr, dev_ldr, test_ldr

def train(train_ldr):
    net.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_ldr):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch_idx, len(train_ldr),
                elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate(dev_ldr):
    net.eval()
    total_loss = 0

    embeds, ids = [], []
    with torch.no_grad():
        for inputs, labels in dev_ldr:
            outputs = net(inputs)

            embeds.append(net.get_embeds(outputs).data)
            ids.append(np.squeeze(labels))
        embeds, ids = np.array(np.concatenate(embeds)), np.array(np.concatenate(ids))
    return average_precision(embeds, ids)

if __name__ == "__main__":
    # Loading data
    print('-' * 89)
    print("Loading data...")
    ntokens, train_ldr, dev_ldr, test_ldr = data_loader()
    print('-' * 89)
    print("Data loaded")
    print('-' * 89)

    net = GatedCNN(out_dims=ntokens, activation=F.tanh)
    net = net.cuda() if torch.cuda.is_available() else net.cpu()
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    best_so_far = 0
    dev_APs = np.empty(MAX_EPOCHS)

    try:
        for epoch in range(1, MAX_EPOCHS+1):
            epoch_start_time = time.time()
            #scheduler.step()
            train(train_ldr)
            dev_ap = evaluate(dev_ldr)
            dev_APs[epoch-1] = dev_ap
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | dev ap {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), dev_ap))
            print('-' * 89)

            #torch.save(net, os.path.join(SAVE_PATH, str(epoch)))
            if dev_ap > best_so_far:
                best_so_far = dev_ap
                torch.save(net, os.path.join(SAVE_PATH, "best"))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('==> Loading best model..')
    best_model = torch.load(os.path.join(SAVE_PATH, "best"))

    best_model.eval()
    total_loss = 0

    embeds, ids = [], []
    with torch.no_grad():
        for inputs, labels in test_ldr:
            outputs = best_model(inputs)

            embeds.append(best_model.get_embeds(outputs).data)
            ids.append(np.squeeze(labels))
        embeds, ids = np.array(np.concatenate(embeds)), np.array(np.concatenate(ids))
    print("test AP: ", average_precision(embeds, ids))
