import argparse
import os

import numpy as np
import torch.nn
import torch.optim as optim
import json

from models import MLP
from oracles import DL2_Oracle
from mimic3_treated import MIMIC3
from mimic3models.metrics import print_metrics_binary
from constraints import *

import time


use_cuda = torch.cuda.is_available()


def oracle_train(args, x_batch, y_batch, oracle):
    n_batch = int(x_batch.size()[0])
    x_batches, y_batches = [], []
    k = n_batch // oracle.constraint.n_tvars
    assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'
    
    for i in range(oracle.constraint.n_tvars):
        x_batches.append(x_batch[i:(i + k)])
        y_batches.append(y_batch[i:(i + k)])

    if oracle.constraint.n_gvars > 0:
        domains = oracle.constraint.get_domains(x_batches, y_batches)
        z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=args.num_iters, args=args)
        _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
    else:
        _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)

    return (dl2_batch_loss, constr_acc)


def train(args, net, oracle, device, train_loader, optimizer, epoch):
    t1 = time.time()
    avg_ce_loss, num_steps= 0, 0 
    predictions, labels = [], []
    ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))

    print('Epoch ', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        num_steps += 1
        x_batch, y_batch = data.to(device), target.to(device)
        
        x_output = net(x_batch)
        ce_batch_loss = ce_loss(x_output, y_batch)

        x_prob = torch.sigmoid(x_output)

        predictions.extend(x_prob.detach().numpy().flatten())
        labels.extend(y_batch.detach().numpy().flatten())
        
        net.eval()
        avg_ce_loss += ce_batch_loss.item()

        if oracle:
            (dl2_batch_loss, constr_acc) = oracle_train(args, x_batch, y_batch, oracle)

        net.train()
        optimizer.zero_grad()
        ce_batch_loss.backward()
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            print('[%d] CE loss: %.3lf, dl2_batch_loss: %.3lf, constr_acc: %.3lf' % (batch_idx, ce_batch_loss.item(), dl2_batch_loss, constr_acc))

    
    print()
    avg_ce_loss /= float(num_steps)
    metrics = print_metrics_binary(labels, predictions, 1)    
    t2 = time.time()
    t = t2 - t1

    print('[Train Set     ] Train acc: %.4f, CE loss: %.3lf, Recall: %.4f, Precision: %.4f, aucroc: %.4f, aucprc: %.4f\n' % (
                metrics['acc'], avg_ce_loss, metrics['rec1'], metrics['prec1'], metrics['auroc'], metrics['auprc']))

    return avg_ce_loss, metrics, t

def test(args, model, oracle, device, test_loader):
    loss = torch.nn.BCEWithLogitsLoss(pos_weight==torch.tensor([args.pos_weight]))
    model.eval()
    avg_ce_loss, num_steps = 0, 0
    predictions, labels = [], []
    
    for data, target in test_loader:
        num_steps += 1
        x_batch, y_batch = data.to(device), target.to(device)

        if oracle:
            (dl2_batch_loss, constr_acc) = oracle_train(args, x_batch, y_batch, oracle)

        x_output = model(x_batch)
        avg_ce_loss += loss(x_output, y_batch).item()

        x_prob = torch.sigmoid(x_output)
        predictions.extend(x_prob.detach().numpy().flatten())
        labels.extend(y_batch.detach().numpy().flatten())

    avg_ce_loss /= float(num_steps)
    metrics = print_metrics_binary(labels, predictions, 1)
    print('[Validation Set] Valid acc: %.4f, CE loss: %.3lf, Recall: %.4f, Precision: %.4f, aucroc: %.4f, aucprc: %.4f\n' % (
                metrics['acc'], avg_ce_loss, metrics['rec1'], metrics['prec1'], metrics['auroc'], metrics['auprc']))

    return avg_ce_loss, metrics


parser = argparse.ArgumentParser(description='Train NN with constraints.')
parser = dl2.add_default_parser_args(parser)
parser.add_argument('--mode', type=str, default="train", help='mode: train or test')
parser.add_argument('--batch-size', type=int, default=64, help='Number of samples in a batch.')
parser.add_argument('--num-iters', type=int, default=50, help='Number of oracle iterations.')
parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs to train for.')
parser.add_argument('--l2', type=int, default=0.01, help='L2 regularizxation.')
parser.add_argument('--pos-weight', type=int, default=3, help='Weight of positive examples.')
parser.add_argument('--grid-search', action=argparse.BooleanOptionalAction)
parser.add_argument('--dl2-weight', type=float, default=0.0, help='Weight of DL2 loss.')
parser.add_argument('--delay', type=int, default=0, help='How many epochs to wait before training with constraints.')
parser.add_argument('--constraint', type=str, required=True, help='the constraint to train with: LipschitzT(L), LipschitzG(eps, L), RobustnessT(eps1, eps2), RobustnessG(eps, delta), CSimiliarityT(), CSimilarityG(), LineSegmentG()')
parser.add_argument('--print-freq', type=int, default=10, help='Print frequency.')
parser.add_argument('--report-dir', type=str, required=True, help='Directory where results should be stored')
parser.add_argument('--network-output', type=str, choices=['logits', 'prob', 'logprob'], default='logits', help='Wether to treat the output of the network as logits, probabilities or log(probabilities) in the constraints.')
args = parser.parse_args()

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

model = MLP(714, 1, 1000, 3).to(device)
       
def Mimic3(eps1, eps2):
    return lambda model, use_cuda, network_output: Mimic3DatasetConstraint(model, eps1, eps2, use_cuda=use_cuda, network_output=network_output)

constraint = eval(args.constraint)(model, use_cuda, network_output=args.network_output)
oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constraint, use_cuda=use_cuda)



if args.mode == 'train':
    mimic_train = MIMIC3(mode='train')
    mimic_val   = MIMIC3(mode='val')
    train_loader = torch.utils.data.DataLoader(mimic_train, shuffle=True, batch_size=args.batch_size, **kwargs)
    val_loader   = torch.utils.data.DataLoader(mimic_val,   shuffle=True, batch_size=args.batch_size, **kwargs)
    
    if not os.path.exists(args.report_dir):
        os.makedirs(args.report_dir)

    if args.grid_search:
        l2_weights = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        pos_weights = [7, 6, 5, 4, 3, 2, 1]
    else:
        l2_weights = [args.l2]
        pos_weights = [args.pos_weight]


    for (l2, pos_weight) in [(l2, pos) for l2 in l2_weights for pos in pos_weights]:
        tstamp = int(time.time())
        args.pos_weight = pos_weight
        report_file = os.path.join(args.report_dir, 'report_l2-%s_weight-%s_%d.json' % (l2, pos_weight, tstamp))

        data_dict = {
            'num_epochs': args.num_epochs,
            'l2_weight': l2,
            'pos_weight': pos_weight,
            'train_avg_loss': [],
            'train_acc': [],
            'train_aucroc': [],
            'train_aucprc': [],
            'epoch_time': [],
            'avg_loss': [],
            'acc': [],
            'aucroc': [],
            'aucprc': [],
        }

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=l2)

        for epoch in range(1, args.num_epochs + 1):
            train_avg_loss, train_metrics, epoch_time = \
                train(args, model, oracle, device, train_loader, optimizer, epoch)
            data_dict['train_avg_loss'].append(train_avg_loss)
            data_dict['train_acc'].append(train_metrics['acc'].item())
            data_dict['train_aucroc'].append(train_metrics['auroc'].item())
            data_dict['train_aucprc'].append(train_metrics['auprc'].item())
            data_dict['epoch_time'].append(epoch_time)

            avg_loss, metrics = test(args, model, oracle, device, val_loader)
            data_dict['avg_loss'].append(avg_loss)
            data_dict['acc'].append(metrics['acc'].item())
            data_dict['aucroc'].append(metrics['auroc'].item())
            data_dict['aucprc'].append(metrics['auprc'].item())
            print(f'Epoch Time [s]: {epoch_time}\n')

        with open(report_file, 'w') as fou:
            json.dump(data_dict, fou, indent=4)

elif args.mode == 'test':
    raise NotImplementedError()

else:
    raise ValueError("Wrong value for args.mode")
