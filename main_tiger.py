from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import pandas as pd
import DataLoader_tiger
from models_tiger import MIL
from train_test_script_tiger import train_basic, test_basic, train_advanced,train_initial_model, init_t_model, MIL_Loss, print_result
import random
from torch.utils.data import DataLoader

import numpy as np

parser = argparse.ArgumentParser(description = 'MIL MNIST-Bags')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=1, help='the size of the batch')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, help='weight decay')
parser.add_argument('--train_sample_size', type=int, default=150, help='number of bags in training set')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--model', type=str, default='attention',
                    help = 'type of aggregation : attention, gated_attention, noisy and, noisy or, ISR, generalized mean, LSE')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


if __name__ == '__main__':
    ### Read from csv file and create train and test dataloader
    csv_file = 'tiger_inst.csv'
    df = pd.read_csv(csv_file)
    dataloader = DataLoader_tiger.TigerDataset(df)

    print('Train the basic MIL model')
    model = MIL(args.model)
    loss = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999), weight_decay = args.reg)
    training_sample = random.sample(range(len(dataloader)), args.train_sample_size)
    testing_sample = np.setdiff1d(range(len(dataloader)),training_sample)
    train = [dataloader[i] for i in training_sample]
    test = [dataloader[i] for i in testing_sample]
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print('Training \n--------')
    train_basic(model, train_loader, loss, optimizer, device, args.epochs)
    print('Testing \n-------')
    test_basic(model, test_loader,device)
    torch.save(model.state_dict(), 'model_0_weights_tiger.pkl')

    print('Train the advanced model')
    num_T=3
    train_loss = dict()
    train_acc = dict()
    # train_initial_model(args, train_loader,test_loader,args.epochs,device)

    for t in range(1, num_T):
        print(f'the model is num {t}')
        model = MIL(args.model)
        model = init_t_model(model, t - 1)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
        loss = MIL_Loss()
        train_loss[t], train_acc[t], model = train_advanced(model, train_loader, loss, optimizer, device, args.epochs)
        torch.save(model.state_dict(), f'model_{t}_weights_tiger.pkl')
        print('\n\n\n\n')
        print(f'train loss for model {t}: {train_loss[t]}')
        print(f'train accuracy for model {t}: {train_acc[t]}')


    # model_path = f'model_2_weights_tiger.pkl'
    # print(f'model 0')
    # print_result(test_loader_, device, model_path)
