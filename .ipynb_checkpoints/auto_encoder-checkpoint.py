import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc
'''Data Prep and Model Evaluation'''
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score


class Read_data:
    def __init__(self,file_pass,label_col_name):
        self.file_pass = file_pass
        self.label_col_name = label_col_name
        self.set_data()
    
    def set_data(self):
        self.df = pd.read_csv(self.file_pass)
    
    def get_label_data(self):
        return self.df[self.label_col_name].values
    
    def extract_feature_data(self):
        return self.df.drop(columns = self.label_col_name)
    
    def get_scalize_feature_data(self):
        scaler = MinMaxScaler()
        featrue_data = self.extract_feature_data()
        scalize_feature_data = scaler.fit_transform(featrue_data)
        
        return scalize_feature_data
    
    def drop_cols(self,list_drop_cols):
        self.df = self.df.drop(columns = list_drop_cols)


class Split_data_train_test:
    def __init__(self,feature,label):
        self.feature = feature
        self.label = label
    
    def get_splited_data(self,test_size = 0.333,shuffle = False):
        X_train, X_test, y_train, y_test = train_test_split(self.feature, self.label, test_size=0.333, shuffle=False)
        
        return X_train, X_test, y_train, y_test


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, X_train, X_valid, epochs=30, batch_size=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train = torch.Tensor(X_train).to(device)
        X_valid = torch.Tensor(X_valid).to(device)

        train_dataset = TensorDataset(X_train, X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                input_data, target = batch
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                valid_output = model(X_valid)
                valid_loss = criterion(valid_output, X_valid)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {valid_loss.item()}")

    def calc_score(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.Tensor(X).to(device)

        with torch.no_grad():
            self.eval()
            output = self(X)
            score = torch.norm(X - output, dim=1).cpu().numpy()

        return score


def plot_MSE(X_valid,y_valid):
    # 検証セットのスコア計算
    score = ae.calc_score(X_valid)
    regular_idx = np.where(y_valid==0)[0]
    anomaly_idx = np.where(y_valid==1)[0]
    # プロット
    plt.plot(regular_idx, score[regular_idx], linestyle='None', marker='.', color='gray', markerfacecolor='None', label='regular')
    plt.plot(anomaly_idx, score[anomaly_idx], linestyle='None', marker='.', color='orange', markerfacecolor='None', label='fraud')
    plt.legend()


from sklearn.metrics import confusion_matrix
def plot_auc(X_data,y_data):
    score = ae.calc_score(X_data)
    threshold = np.linspace(0, 0.2, 100)
    cms = [confusion_matrix(y_data, np.where(score >= th, 1, 0)).ravel() for th in threshold]
    accuracy = np.array([(tp+tn) / (tp+fp+fn+tn) for tn, fp, fn, tp in cms])
    fn_rate = np.array([fn / (tp+fn) for tn, fp, fn, tp in cms])
    fp_rate = np.array([fp / (tn+fp) for tn, fp, fn, tp in cms])
    plt.plot(threshold, accuracy, label='accuracy', linestyle='dashed')
    plt.plot(threshold, fn_rate, label='fn rate')
    plt.plot(threshold, fp_rate, label='fp rate')
    index = np.argmin(np.abs(fn_rate-fp_rate))     # 偽陽性率、偽陰性率が同程度になる閾値のインデックス
    plt.vlines(threshold[index], 0, 1.0, linestyle='dashed')
    plt.legend()
    cm = confusion_matrix(y_test, np.where(score >= threshold[index], 1, 0))
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print('accuracy: ', (tp+tn) / (tp+fp+fn+tn))
    print('fn_rate: ', fn / (tp+fn))
    print('fp_rate: ', fp / (tn+fp))
        