import matplotlib.pyplot as plt 
import numpy as np
import random
import math

RATIO = 0.8
N = 610
M = 9742
D = 60
lr = 0.005
EPOCH = 200
sigma_0 = 1
sigma_U = 0.001
sigma_V = 0.0001

def SGD(train,test):
    U = np.random.normal(0, sigma_0, (N, D))
    V = np.random.normal(0, sigma_0, (M, D))
    rmse=[]
    loss=[]
    total_u = total_v = 0
    for ste in range(EPOCH):
        los=0.0
        for u,i,r in train:
            predict = sigmoid(np.dot(U[u],V[i]))
            e=r-predict 
            square_old_u = sum(np.square(U[u]))
            square_old_v = sum(np.square(V[i]))
            old_u = np.array(U[u])
            U[u]+=lr*(e*predict*(1-predict)*V[i]-sigma_U*U[u])
            V[i]+=lr*(e*predict*(1-predict)*old_u-sigma_V*V[i])
            los=los+0.5*(e**2+sigma_U*square_old_u+sigma_V*square_old_v)
            total_u += square_old_u
            total_v += square_old_v
        loss.append(los/len(train))
        rms=RMSE(U,V,test)
        rmse.append(rms)
        if ste%2==0:
            print(ste/10)
    return loss,rmse

def sigmoid(value):
    return (math.e**value)/(1+math.e**value)

def RMSE(U,V,test):
    count=len(test)
    sum_rmse=0.0
    for t in test:
        u=t[0]
        i=t[1]
        r=t[2]
        pr=sigmoid(np.dot(U[u],V[i]))
        sum_rmse+=np.square(r-pr)
    rmse=np.sqrt(sum_rmse/count)
    return rmse


def Load_data(filedir, mapping_id):
    data = []
    f = open(filedir)
    for line in f.readlines():
        user_id, movie_id, rating = line.split(",")
        data.append([int(user_id)-1, mapping_id[int(movie_id)], (float(rating)-0.5)/4.5])
    f.close()
    np.random.shuffle(data)
    train=data[0:int(len(data)*RATIO)]
    test=data[int(len(data)*RATIO):]
    return train,test

def read_id(fileloc):
    mapping_id = {}
    file = open(fileloc)
    movie_id = 0
    for line in file.readlines():
        fake_id = line.split()[0]
        mapping_id[int(fake_id)] = movie_id
        movie_id += 1
    file.close()
    return mapping_id

def plotting(list_, name):
    fig1=plt.figure()
    plt.plot(list(range(len(list_))), list_, color='g')
    plt.title('Convergence curve')
    plt.xlabel('Iterations')
    plt.ylabel(name)
    plt.show()


def main():
    movie_dir="movies.csv"
    rating_dir = "ratings.csv"
    mapping_id = read_id(movie_dir)
    train,test=Load_data(rating_dir,mapping_id)
    loss,rmse =SGD(train,test)
    plotting(loss, 'loss')
    plotting(rmse, 'rmse')
    
         
if __name__ == '__main__': 
    main()