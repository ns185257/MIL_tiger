import torch
from torch.autograd import Variable
import torch.nn as nn
import itertools
import numpy as np
import torch.optim as optim
from models_tiger import MIL

def calculate_classification_error(Y, Y_hat):
    """
    Calculates the classification error.
    Parameters
    ----------
    Y : GT labels
    Y_hat : Predicted labels
    Returns
    -------
    error : classification error
    Y_hat : predicted labels
    """

    Y = Y.float()
    error = 1. - torch.tensor(Y_hat).eq(Y).cpu().float().mean().item()

    return error

def calculate_objective(Y, Y_prob):
    """
    Calculates the loss.
    Parameters
    ----------
    Y : GT labels
    Y_prob : Predicted probabilities
    Returns
    -------
    neg_log_likelihood : The negative log-likelihood
    """

    # Convert labels to float
    Y = Y.float()
    # Clip the predicted probabilities
    Y_prob = torch.clamp(Y_prob, min = 1e-5, max = 1. - 1e-5)
    # Binary cross entropy
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

    return neg_log_likelihood

def train_basic(model, train_loader, loss_fn, optimizer, device, epochs):
    """
    Trains the given model with given loss function and optimizer
    Parameters
    ----------
    model        : Model to be trained
    train_loader : Data loader
    loss_fn      : Loss function
    optimizer    : Optimizer
    device       : cuda or cpu
    epochs       : No. of iterations to be trained
    """
    model.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_error = 0.
        train_acc =0.
        for X, Y, _ in train_loader:
            X, Y = X.to(device), Y.to(device) #X = [batch, bag_len,87]   Y = label
            X, Y = Variable(X), Variable(Y) #

            # reset gradients
            optimizer.zero_grad()
            # forward pass
            Y_prob, Y_hat = model(X.float())
            # calculate loss
            loss = loss_fn(Y_prob, Y.float())
            # update total loss
            train_loss += loss.item()
            error = calculate_classification_error(Y, Y_hat)
            # update total error
            train_error += error
            # backward pass
            loss.backward()
            # update the parameters
            optimizer.step()
            train_acc += 1 if torch.tensor(Y_hat).item()==Y.float().item() else 0

        # calculate average loss and error
        train_loss /= len(train_loader)
        train_error /= len(train_loader)
        train_acc /= len(train_loader)

        print('Epoch {}/{} : Loss = {:.4f}, Error = {:.4f}, Accuracy = {:.4f}'.format(epoch+1, epochs, train_loss, train_error, train_acc))

def test_basic(model, test_loader, device):
    """
    Tests the model
    Parameters
    ----------
    model       : The model to be tested
    test_loader : Data loader
    loss_fn     : The loss function
    device      : cuda or cpu
    """
    model.eval()
    test_error = 0.
    test_acc = 0.
    for X, Y, _ in test_loader:
        X, Y = X.to(device), Y.to(device)
        X, Y = Variable(X), Variable(Y)
        # forward pass
        Y_prob, Y_hat = model(X.float())
        # compute classification error
        error = calculate_classification_error(Y, Y_hat)
        # update total error
        test_error += error
        # y_pred = torch.round(Y_hat)
        test_acc += 1 if torch.tensor(Y_hat).item()==Y.float().item() else 0

    test_error /= len(test_loader)
    test_acc /= len(test_loader)

    print('Error = {:.4f},  Accuracy = {:.4f}'.format( test_error,test_acc))

def init_t_model(v_model, t ):
    classname = v_model.__class__.__name__
    model_dict = torch.load(f'models/model_{t}_weights_tiger.pkl',
                            map_location=lambda storage, loc: storage)
    if classname.find('Conv') != -1:
        nn.init.normal_(model_dict.weight, 0.0, 0.02)

    v_model.load_state_dict(model_dict)
    return v_model

class MIL_Loss(torch.nn.Module):

    def __init__(self):
        super(MIL_Loss, self).__init__()

    def forward(self, Y,y_prob_list, bag_size, w_dict):
        A = list(itertools.product([0, 1], repeat=bag_size))
        A.remove(tuple([0] * bag_size))
        A_ops = [tuple(np.logical_not(a).astype(int)) for a in A]
        y_prob_log = torch.log(y_prob_list)
        y_prob_ops = 1 - y_prob_list
        y_prob_log_ops = torch.log(y_prob_ops)
        loss_sum = 0

        for a, a_ops in zip(A,A_ops):
            torch.tensor(a)
            torch.tensor(a_ops)
            if Y == 1:
                if len(y_prob_log) == 1:
                    loss_sum += (torch.dot(y_prob_log, torch.tensor(a).float()) +
                                 torch.dot(y_prob_log_ops, torch.tensor(a_ops).float())) * w_dict[a]
                else:
                    loss_sum += (torch.dot(y_prob_log.squeeze(0), torch.tensor(a).float()) +
                                 torch.dot(y_prob_log_ops.squeeze(0), torch.tensor(a_ops).float())) * w_dict[a]
            else:
                loss_sum += torch.sum(y_prob_log_ops)
        return loss_sum

def calc_w(y_prob_list):
    """
    :param y_prob_list: list of prob that y_i=1 for bag X
    :return: dict of weights => W[(1,0,1)] = w
    """
    A = list(itertools.product([0, 1], repeat=len(y_prob_list)))
    A.remove(tuple([0]*len(y_prob_list)))

    W = dict()
    for a in A:
        w1 = [y_prob for a_, y_prob in zip(a,y_prob_list) if a_==1]
        w2 = [(1-y_prob) for a_, y_prob in zip(a,y_prob_list) if a_==0]
        w = w1+w2
        w = np.prod([item.item() for item in w])
        norm = 1 - np.prod([1-y_prob.detach().numpy() for y_prob in y_prob_list])
        W[(a)] = w/norm
    return W

def train_advanced(model, train_loader, loss_fn, optimizer, device, epochs):
    train_loss_epoch, train_acc_epoch = list(), list()
    model.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_acc =0.
        for idx, (x, y, _) in enumerate(train_loader):
            y_prob_list = []
            y_hat_list = []
            for X in x[0]:
                X = X.to(device) #X = [87]   Y = label
                X = Variable(X, requires_grad=True)
                X = X.unsqueeze(0)
                X = X.unsqueeze(0)
                Y_prob, Y_hat = model(X.float())
                y_prob_list.append(Y_prob)
                y_hat_list.append(Y_hat)
            w_dict = calc_w(y_prob_list)
            optimizer.zero_grad()
            loss = loss_fn(y,torch.cat(y_prob_list), x.shape[1], w_dict)
            train_loss += (loss.item()/x.shape[1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0)
            optimizer.step()
            Y_hat_bag = 1 if 1 in y_hat_list else -1
            train_acc += 1 if Y_hat_bag==y.float().item() else 0
            for j,p in enumerate(model.parameters()):
                a= torch.isnan(p.grad)
                if True in a:
                    print('##########   ERROR    ##########')
                    print(f' bag number {idx}')
                    break
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)
    return train_loss_epoch, train_acc_epoch, model

def test_advanced(model, test_loader1, device):
    model.eval()
    test_acc = 0.
    for x,y,_ in test_loader1:
        y_hat_list = []
        for X in x[0]:
            X = X.to(device)
            X= Variable(X)
            X = X.unsqueeze(0)
            X = X.unsqueeze(0)
            Y_prob, Y_hat = model(X.float())
            y_hat_list.append(Y_hat)

        Y_hat_bag = 1 if 1 in y_hat_list else 0
        test_acc += 1 if Y_hat_bag == y.float().item() else 0

    test_acc /= len(test_loader1)
    print(' Accuracy = {:.4f}'.format(test_acc))

def train_initial_model(args, train_loader,test_loader,epochs,device):
    model = MIL(args.model)
    loss = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999), weight_decay = args.reg)
    train_basic(model, train_loader, loss, optimizer, device, epochs=epochs)
    torch.save(model.state_dict(), 'model_0_weights.pkl')
    test_basic(model, test_loader, device)

def print_result(test_loader,device, model_path):
    model = MIL()
    model.load_state_dict(torch.load(model_path))
    print('The accuracy calculated on each bag:')
    test_basic(model, test_loader, device)
    print('The accuracy calculated on item in each bag:')
    test_advanced(model, test_loader, device)
