import os 
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random

import torch
from torch.optim import Adam, SGD

from CGCNN import CGCNN
from data import Get_feature_and_Load_Data

from torch_geometric.loader import DataLoader

# fix random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    

### Model train
def run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, optimizer, epochs, batch_size, 
        vt_batch_size, lr, weight_decay, save_dir='models/', disable_tqdm=False):     

    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

    best_valid = float('inf')
    test_valid = float('inf')
        
    if save_dir != '':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    start_epoch = 1
    
    train_mae_list = []
    valid_mae_list = []
    test_mae_list = []
    
    for epoch in range(start_epoch, epochs + 1):
        print("=====Epoch {}".format(epoch), flush=True)
        t_start = time.perf_counter()
        
        train_mae = train(model, optimizer, train_loader, loss_func, device, disable_tqdm)
        valid_mae = val(model, valid_loader, device, disable_tqdm)
        test_mae = val(model, test_loader, device, disable_tqdm)

        train_mae_list.append(train_mae)
        valid_mae_list.append(valid_mae)
        test_mae_list.append(test_mae)       
        
        if valid_mae < best_valid:
            best_valid = valid_mae
            test_valid = test_mae
            if save_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(save_dir, 'Best_valid_model.pt'))

        t_end = time.perf_counter()
        print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae, 'Best valid': best_valid, 'Test@ best valid': test_valid, 'Duration': t_end-t_start})
    
    # draw a train, valid, test plot vs epoch   
    def plot_metrics(train_mae_list, valid_mae_list, test_mae_list):
        plt.figure(figsize=(10, 6))
        plt.plot(train_mae_list, label='Train MAE')
        plt.plot(valid_mae_list, label='Validation MAE')
        plt.plot(test_mae_list, label='Test MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training, Validation, and Test MAE vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('train_test_model.png')
        
    # Call the plot function after the training loop
    plot_metrics(train_mae_list, valid_mae_list, test_mae_list)

    print(f'Best validation MAE so far: {best_valid}')
    print(f'Test MAE when got best validation result: {test_valid}')
    
        
def train(model, optimizer, train_loader, loss_func, device, disable_tqdm):  
    model.train()
    loss_accum = 0
    for step, batch_data in enumerate(tqdm(train_loader, disable=disable_tqdm)):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        loss = loss_func(out, batch_data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / (step + 1)

def val(model, data_loader, device, disable_tqdm):   
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    
    for step, batch_data in enumerate(tqdm(data_loader, disable=disable_tqdm)):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        preds = torch.cat([preds, out.detach_()], dim=0)
        targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

    return torch.mean(torch.abs(preds - targets)).cpu().item()

parser = argparse.ArgumentParser(description='CGCNN')
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--node_embedding', type=int, default=64)
parser.add_argument('--num_conv_layer', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--vt_batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=150)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--save_dir', type=str, default='models/')
parser.add_argument('--disable_tqdm', default=False, action='store_true')
parser.add_argument('--csv_name', type=str, default='mp_fm.csv', help='csv file name')

args = parser.parse_args()

# load data
data = Get_feature_and_Load_Data(csv_name=args.csv_name)
data_list = data.get_atom_and_bond_fea_and_load()

# shuffle data 
new_data_list=shuffle(data_list,random_state=args.seed)

# Calculate split indices
total_len = len(data_list)
train_len = int(0.8 * total_len)
valid_len = int(0.1 * total_len)
# The remaining data will be used for testing

# Split the data
train_dataset = new_data_list[:train_len]
valid_dataset = new_data_list[train_len:train_len+valid_len]
test_dataset = new_data_list[train_len+valid_len:]
# print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

model = CGCNN(node_fea_len=args.node_embedding,edge_fea_len=41,orig_node_fea_len=92,
              num_conv_layer=args.num_conv_layer, hidden_layer_len=args.hidden_channels)

loss_func = torch.nn.MSELoss()

if args.optim == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
model.to(device)

run(device=device, 
    train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, 
    model=model, loss_func=loss_func, optimizer=optimizer,
    epochs=args.epochs, batch_size=args.batch_size, vt_batch_size=args.batch_size, 
    lr=args.lr, weight_decay=args.weight_decay, save_dir=args.save_dir)


##### load trained model to predict and evaluate #####
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

def rmse(y_true, y_pred):  
    return mean_squared_error(y_true, y_pred,squared=False)

def mse(y_ture,y_pred):
    return mean_squared_error(y_ture,y_pred)

def mae(y_ture,y_pred):
    return mean_absolute_error(y_ture,y_pred)

def get_predictions_and_targets(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_id = []
    
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        all_preds.extend(out.cpu().numpy())
        all_targets.extend(batch_data.y.cpu().numpy())
        all_id.extend(batch_data.id)
        
    return all_preds, all_targets, all_id

checkpoint = torch.load('models/Best_valid_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, args.vt_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.vt_batch_size, shuffle=False)
train_preds, train_targets, train_id = get_predictions_and_targets(model, train_loader, device)
test_preds, test_targets, test_id = get_predictions_and_targets(model, test_loader, device)
train_preds=[i[0] for i in train_preds]
test_preds=[i[0] for i in test_preds]

#save to csv
train_data_csv = pd.DataFrame({'id':train_id,
                               'target':train_targets,
                               'predict':train_preds})
test_data_csv = pd.DataFrame({'id':test_id,
                               'target':test_targets,
                               'predict':test_preds})

train_data_csv.to_csv('train_data.csv',index=False)
test_data_csv.to_csv('test_data.csv',index=False)

print()
print('train r2_score=',r2_score(train_targets, train_preds))
print('test r2_score=',r2_score(test_targets,  test_preds))
print('rmse_test=',rmse(test_targets,  test_preds))
print('mse_test=',mse(test_targets,  test_preds))
print('mae_test=',mae(test_targets,  test_preds))
print('rmse_train=',rmse(train_targets, train_preds))
print('mse_train=',mse(train_targets, train_preds))
print('mae_train=',mae(train_targets, train_preds))

plt.figure(figsize=(6,5))
plt.scatter(test_targets,  test_preds,label='test data')
plt.scatter(train_targets, train_preds,label='train data')
target_value=test_targets+train_targets
plt.plot(target_value,target_value,'-k')
plt.legend()
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.savefig('R2.png')