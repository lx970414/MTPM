import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics

from typing import Any, Optional, Tuple
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
# from processing_data import pro_data_main
import warnings
warnings.filterwarnings("ignore")


batch_size = 128
initial_learning_rate = 0.01  
epochs = 200 
repeats=10 

def divide_infor_label(data):
    link_label = data[:, 0]  
    infor = data[:, 1:]  
    return link_label, infor


def divide_network_edge(data):
    network_label = data[:, :, 0]  
    edge = data[:, :, 1:]  
    return network_label, edge


def get_train_valid_test(dataset,Proportion):
    data = pd.read_csv('node2vec/' + dataset + '.txt', header=None)
    data = np.array(data)
    np.random.shuffle(data)

    link_label, infor = divide_infor_label(data)
    
    train_infor, test_infor, train_label, test_label = train_test_split(infor, link_label, test_size=1-Proportion)
 
    train_infor, valid_infor, train_label, valid_label= train_test_split(train_infor, train_label, test_size=0.1)
    print("train counter", sorted(Counter(train_label).items()))
    print("valid counter: ", sorted(Counter(valid_label).items()))
    print("test counter: ", sorted(Counter(test_label).items()))

    rus = RandomUnderSampler(random_state=0, replacement=True)
    train_infor, train_label = rus.fit_resample(train_infor, train_label)
    print("train under sampling results: ", sorted(Counter(train_label).items()))
    train_infor = torch.from_numpy(train_infor).unsqueeze(dim=1).float()
    train_label = torch.from_numpy(train_label).unsqueeze(dim=1).float()
    train_set = TensorDataset(train_infor, train_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    valid_infor = torch.from_numpy(valid_infor).unsqueeze(dim=1).float()
    valid_label = torch.from_numpy(valid_label).unsqueeze(dim=1).float()
    valid_set = TensorDataset(valid_infor, valid_label)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    
    test_infor = torch.from_numpy(test_infor).unsqueeze(dim=1).float()
    test_label = torch.from_numpy(test_label).unsqueeze(dim=1).float()
    test_set = TensorDataset(test_infor, test_label)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)



class LP(nn.Module):
    def __init__(self, in_dim , network_numbers):
        super(LP, self).__init__()

        self.generality_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=100, kernel_size=1),
            nn.ReLU())

        self.target_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=100, kernel_size=1),
            nn.ReLU())

        self.weight_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim+1, out_channels=100, kernel_size=1),
            nn.ReLU())

        self.weight_softmax = nn.Sequential(
            nn.Linear(100, 2),
            nn.Softmax(dim=1))

        self.link_classifier = nn.Sequential(
            nn.Linear(100, 2),
            nn.Softmax(dim=1))

        self.network_classifier = nn.Sequential(
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Linear(60, network_numbers),
            nn.Softmax(dim=1))

    def forward(self, edge_embbing, weight_input, coeff=1):
        
        edge_embbing = edge_embbing.permute(0, 2, 1)

        generality_feature = self.generality_conv(edge_embbing)
        generality_feature = generality_feature.view(generality_feature.size(0), -1)


        
        target_feature = self.target_conv(edge_embbing)
        target_feature = target_feature.view(target_feature.size(0), -1)
       
        weight_input = weight_input.permute(0, 2, 1) 
        weight_out = self.weight_conv(weight_input)
        weight_out = weight_out.view(weight_out.size(0), -1)
        weight_out = self.weight_softmax(weight_out)
        
        feature = torch.zeros_like(target_feature)
        for i in range(feature.shape[0]):
            feature[i] = generality_feature[i] * weight_out[i][0] + target_feature[i] * weight_out[i][1]
        
        link_output = self.link_classifier(feature)
        
        reverse_feature = grad_reverse(generality_feature, coeff)
        network_output = self.network_classifier(reverse_feature)
        return link_output, network_output


def get_pred(out):
    out = out[:, 1]
    one = torch.ones_like(out)
    zero = torch.zeros_like(out)
    out = torch.where(out >= 0.5, one, zero)
    return out

def get_acc(out, label):
    out = get_pred(out)
    accuracy = (out == label).float().mean()
    return accuracy

def train__Model(dataset, train_loader, valid_loader, model, criterion):

    model_path = 'result/' + dataset + '_model/'
    if os.path.exists(model_path):  
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    
    best_valid_dir = ''
    best_valid_acc = 0
    for epoch in range(epochs+1):
        p = epoch / epochs
        learning_rate = initial_learning_rate / pow((1 + 10 * p), 0.75)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        
        model.train()
        loss_vec = []
        acc_vec = []
        for data in train_loader:
            # infor+label
            infor, link_label = data
            network_label, edge = divide_network_edge(infor)
            # edge,link_label,network_label
            if torch.cuda.is_available():
                infor = infor.cuda()
                edge = edge.cuda()
                link_label = link_label.cuda()
                network_label = network_label.cuda()
            infor = Variable(infor)
            edge = Variable(edge)
            link_label = Variable(link_label)
            network_label = Variable(network_label)
            
            link_out, network_out = model(edge,infor)
            link_loss = criterion(link_out, link_label.squeeze(1).long())
            network_loss = criterion(network_out, network_label.squeeze(1).long())
            loss = link_loss + network_loss
            loss_vec.append(loss.detach().cpu().numpy())
            acc = get_acc(link_out, link_label.squeeze(1).long())
            acc_vec.append(acc.detach().cpu().numpy())
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        loss = np.mean(loss_vec)
        acc = np.mean(acc_vec)

        
        model.eval()
        valid_acc_vec = []
        for data in valid_loader:
            # infor+label
            infor, link_label = data
            _, edge = divide_network_edge(infor)
            if torch.cuda.is_available():
                with torch.no_grad():
                    infor = Variable(infor).cuda()
                    edge = Variable(edge).cuda()
                    link_label = Variable(link_label).cuda()
            else:
                with torch.no_grad():
                    infor = Variable(infor)
                    edge = Variable(edge)
                    link_label = Variable(link_label)
            link_out, _ = model(edge, infor)
            valid_acc = get_acc(link_out, link_label.squeeze(1).long())
            valid_acc_vec.append(valid_acc.detach().cpu().numpy())
        valid_acc = np.mean(valid_acc_vec)

        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_dir = model_path + 'model' + str(epoch) + '.pkl'
            torch.save(model.state_dict(), best_valid_dir)

        if epoch % 10 == 0:
             print('Model Epoch: [{}/{}], learning rate:{:.6f}, train loss:{:.4f}, train acc:{:.4f}, valid acc:{:.4f}'.format(epoch, epochs, learning_rate, loss, acc, valid_acc))

    return best_valid_dir


def test__Model(test_loader, lp_model, best_valid_dir):

    lp_model.load_state_dict(torch.load(best_valid_dir))
    lp_model.eval()

    acc_vec = []
    precision_vec=[]
    recall_vec = []
    f1_vec = []
    auc_vec = []
    
    for data in test_loader:
        
        infor, link_label = data
        _, edge = divide_network_edge(infor)
        if torch.cuda.is_available():
            with torch.no_grad():
                infor = Variable(infor).cuda()
                edge = Variable(edge).cuda()
                link_label = Variable(link_label).cuda()
        else:
            with torch.no_grad():
                infor = Variable(infor)
                edge = Variable(edge)
                link_label = Variable(link_label)

        lp_out, _ = lp_model(edge, infor)
        pred = get_pred(lp_out).cpu()
        link_label = link_label.squeeze(1).long().cpu()
        acc = (pred == link_label).float().mean()
        acc_vec.append(acc.detach().cpu().numpy())
        
        precision = metrics.precision_score(link_label, pred, average='weighted')
        recall = metrics.recall_score(link_label, pred, average='weighted')
        f1 = metrics.f1_score(link_label, pred, average='weighted')
        precision_vec.append(precision)
        recall_vec.append(recall)
        f1_vec.append(f1)
        try:
            auc = metrics.roc_auc_score(link_label, pred)
            auc_vec.append(auc)
        except ValueError:  
            pass

    acc = np.mean(acc_vec)
    precision=np.mean(precision_vec)
    recall = np.mean(recall_vec)
    f1 = np.mean(f1_vec)
    auc = np.mean(auc_vec)
    
    return acc,precision,recall,f1,auc

def run__model(dataset, train_loader, valid_loader, test_loader, network_numbers):
    lp_model = LP(in_dim=64, network_numbers=network_numbers)
    if torch.cuda.is_available():
        lp_model = lp_model.cuda()
    criterion = nn.CrossEntropyLoss()
    best_valid_dir = train__Model(dataset, train_loader, valid_loader, lp_model, criterion)
    
    acc,precision,recall,f1,auc = test__Model(test_loader, lp_model, best_valid_dir)
    return acc,precision,recall,f1,auc


if __name__ == "__main__":

    resultpath = 'result/'
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    outfile = open('result/out.txt', 'w', encoding='utf-8')

    datasets = ['Aarhus', 'Enron', 'Kapferer', 'London','TF']
    
    Proportion_list = [0.8] 
    pro={'Aarhus':5,'Enron':2, 'EUAir':37, 'Kapferer':4, 'London':3, 'TF':2}

    for dataset in datasets:
        write_infor = '\ndataset:' + dataset + ' epochs:{}\n'.format(epochs)
        print(write_infor)
        outfile.write(write_infor)
        #network_number = pro_data_main(dataset)
        network_number =pro[dataset]
        for Proportion in Proportion_list:
            acc_t=[]
            precision_t=[]
            recall_t=[]
            f1_t=[]
            auc_t=[]
            for repeat in range(repeats):
                train_loader, valid_loader, test_loader = get_train_valid_test(dataset, Proportion)
                acc,precision,recall,f1,auc = run__model(dataset,train_loader, valid_loader, test_loader, network_number)
                acc_t.append(acc)
                precision_t.append(precision)
                recall_t.append(recall)
                f1_t.append(f1)
                auc_t.append(auc)
                write_infor = 'repeat:{}, acc:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}, auc:{:.4f}\n'.format(
                    repeat+1, acc, precision, recall, f1, auc)
                print(write_infor)
                outfile.write(write_infor)
            acc = np.mean(acc_t)
            precision = np.mean(precision_t)
            recall = np.mean(recall_t)
            f1 = np.mean(f1_t)
            auc = np.mean(auc_t)
            write_infor = 'average,  acc:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}, auc:{:.4f}\n'.format(
                acc,precision,recall,f1,auc)
            print(write_infor)
            outfile.write(write_infor)
        outfile.write('\n---------------------------------------------------------------------------------------------------\n')
    outfile.close()

