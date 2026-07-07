import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml 

mnist = fetch_openml('mnist_784', version=1, as_frame=False) 

Y = mnist.target.astype(np.int64) 

X = mnist.data.astype(np.float32) 
X = (X - np.mean(X))/np.std(X)

X_train = X[:40000]
Y_train = Y[:40000]

X_val = X[40000:60000]
Y_val = Y[40000:60000]

X_test = X[60000:]
Y_test = Y[60000:]

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train)

X_test = torch.tensor(X_test)
Y_test = torch.tensor(Y_test)

X_val = torch.tensor(X_val)
Y_val = torch.tensor(Y_val)

train_dataset = TensorDataset(X_train,Y_train)
val_dataset = TensorDataset(X_val,Y_val)
test_dataset = TensorDataset(X_test,Y_test)

num = int(input("Enter number of layers :")) 
if num < 2:
    raise ValueError 
n,test_accuracy,train_loss = [],[],[]
for ne in range(num - 1): 
    neurons = int(input(f"Enter number of neurons for layer {ne + 1} : "))
    n.append(neurons)

class MLP(nn.Module):
    def __init__(self,num,n):
        super().__init__()
        self.num,self.n,self.p = num,n,nn.ModuleList()
        for k in range(self.num):
            if k == 0:
                a = nn.Linear(X_train.shape[1],self.n[k])
                self.p.append(a)
            elif k < self.num - 1:
                a = nn.Linear(self.n[k-1],self.n[k])
                self.p.append(a)
            elif k == self.num - 1:
                a = nn.Linear(self.n[k-1],10)
                self.p.append(a)
        self.relu = nn.ReLU()
        self.output = nn.Softmax(dim = 1)
        self.loss = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(p = 0.2)

    
    def forward(self,data):
        self.data = data
        for j in range(self.num):
            if j < num - 1:
                self.data = self.p[j](self.data)
                self.data = self.relu(self.data)
                self.data = self.drop(self.data)
            else:
                self.data = self.p[j](self.data)
        return self.data

model = MLP(num,n)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01) 
loaded_trainer = DataLoader(dataset = train_dataset,shuffle=True,batch_size=32)
loaded_tester = DataLoader(dataset=test_dataset,shuffle=True,batch_size=32)
loaded_validation = DataLoader(dataset=val_dataset,shuffle=True,batch_size=32)
for q in range(10):
    batch = 0
    each_batch_loss = 0
    for X_t,Y_t in loaded_trainer:
        model.train()
        result = model(X_t)
        loss = model.loss(result,Y_t)
        each_batch_loss += loss 
        batch += 1 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
    
    t_loss = each_batch_loss/batch
    train_loss.append(t_loss.detach())
    correct = 0
    for x_v,y_v in loaded_validation:
        with torch.no_grad():
            model.eval()
            z = model(x_v)
        
        final = model.output(z)
        pred = torch.argmax(final,axis = 1)
        correct += torch.sum(pred == y_v)
    
    accuracy = correct/X_val.shape[0]
    test_accuracy.append(accuracy.detach())

correct = 0
for x_t,y_t in loaded_tester: 
    with torch.no_grad():
        model.eval()
        z = model(x_t)

    final = model.output(z)
    pred = torch.argmax(final,axis = 1)
    correct += torch.sum(pred == y_t)
accuracy = correct/X_test.shape[0]

plt.figure()
plt.plot(train_loss)
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(test_accuracy)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

print(accuracy) 
torch.save(model.state_dict(),"model.pth")
