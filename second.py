import numpy as np
from sklearn.datasets import fetch_openml 
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version=1, as_frame=False) 

X = mnist.data.astype(np.float32) 
Y = mnist.target.astype(np.int64) 

X = (X - np.mean(X))/np.std(X)


X_train = X[:60000] 
Y_train = Y[:60000]

X_test = X[60000:]
Y_test = Y[60000:]


def init(dims,n): 
    s = np.sqrt(2.0/dims)
    w = np.random.randn(dims,n) * s
    b = np.zeros((1,n))
    return w,b

def forward(input,w,b): 
    return input@w + b

def relu_a(input):
    return np.maximum(0,input)

def softmax_a(input): 
    shifted = input - np.max(input,axis = 1,keepdims=True)
    exp_e = np.exp(shifted)
    return exp_e/np.sum(exp_e,axis = 1,keepdims=True)

def Loss(pre,real):
    N = pre.shape[0]
    return -np.mean(np.log(pre[np.arange(N),real] + 1e-15))

w,b,A,z = [],[],[],[]  
train_losses = []
test_accracy = []

def g_l_z(pre,real): 
    dl_z = pre.copy()
    dl_z[np.arange(pre.shape[0]),real] -= 1
    dl_z/=pre.shape[0]
    return dl_z

def g_l_w(dl_z,input): 
    return input.T@dl_z

def g_l_b(dl_z):
    return np.sum(dl_z,axis = 0,keepdims=True)

def g_l_a(dl_z,W): 
    return dl_z @ W.T

def g_l_z2(dl_a,linear): 
    return dl_a * (linear > 0)

def update(W,B,dl_w,dl_b,lr): 
    W = W - dl_w * lr
    B = B - dl_b * lr
    return W,B

num = int(input("Enter number of layers :")) 
if num < 2:
    raise print("Enter value greater than 1")

lr = float(input("Enter learning rate : "))

n = []
for ne in range(num - 1): 
    neurons = int(input(f"Enter number of neurons for layer {ne + 1} : "))
    n.append(neurons)

w,b,z,A = [],[],[],[]

for k in range(num): 
    if k == 0:
        we,bi = init(X_train.shape[1],n[k])
        w.append(we),b.append(bi)
    elif k < num - 1:
        we,bi = init(n[k - 1],n[k])
        w.append(we),b.append(bi)
    elif k == num - 1:
        we,bi = init(n[k - 1],10)
        w.append(we),b.append(bi)

for q in range(10):
    perm = np.random.permutation(X_train.shape[0]) 
    X_train = X_train[perm]
    Y_train = Y_train[perm]
    batches = 0
    epoch_loss = 0
    for s in range(0,X_train.shape[0],32):
        X_t = X_train[s:s+32]
        Y_t = Y_train[s:s+32]

        A,z = [],[]

        for j in range(num): 
            if j == 0:
                li = forward(X_t,w[j],b[j])
                a = relu_a(li)
                z.append(li),A.append(a)
            elif j < num - 1:
                li = forward(A[j - 1],w[j],b[j])
                a = relu_a(li)
                z.append(li),A.append(a)
            elif j == num - 1:
                li = forward(A[j - 1],w[j],b[j])
                z.append(li)
                so = softmax_a(z[j])
                batches+=1
                epoch_loss = Loss(so,Y_t)

        dl_w,dl_b = [],[]
        for l in reversed(range(num)): 
                if l == num - 1:
                    dl_z = g_l_z(so,Y_t)
                    l_w = g_l_w(dl_z,A[l - 1])
                    l_b = g_l_b(dl_z)
                    dl_a = g_l_a(dl_z,w[l])
                    dl_w.append(l_w),dl_b.append(l_b)
                elif l > 0:
                    dl_z2 = g_l_z2(dl_a,z[l])
                    l_w = g_l_w(dl_z2,A[l - 1])
                    l_b = g_l_b(dl_z2)
                    dl_a = g_l_a(dl_z2,w[l])
                    dl_w.append(l_w),dl_b.append(l_b)
                elif l == 0:
                    dl_z2 = g_l_z2(dl_a,z[l])
                    l_w = g_l_w(dl_z2,X_t)
                    l_b = g_l_b(dl_z2)
                    dl_w.append(l_w),dl_b.append(l_b)
        
        dl_w = dl_w[::-1]
        dl_b = dl_b[::-1]

        for m in range(num): 
            w[m],b[m] = update(w[m],b[m],dl_w[m],dl_b[m],lr)
    t_loss = epoch_loss/batches
    train_losses.append(t_loss)
    correct = 0

    for s in range(0, X_test.shape[0], 32):
        X_t = X_test[s:s+32]
        Y_t = Y_test[s:s+32]

        A = X_t

        for j in range(num):
            Z = A @ w[j] + b[j]
            if j < num - 1:
                A = np.maximum(0, Z)
            else:
                A = Z

        shifted = A - np.max(A, axis=1, keepdims=True)
        exp_e = np.exp(shifted)
        probs = exp_e / np.sum(exp_e, axis=1, keepdims=True)

        pred = np.argmax(probs, axis=1)
        correct += np.sum(pred == Y_t)

    accuracy = correct / X_test.shape[0]
    test_accracy.append(accuracy)
    print(epoch_loss)
    print(accuracy)

    print("\n")

plt.figure()
plt.plot(train_losses)
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(test_accracy)
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()