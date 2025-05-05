import matplotlib.pyplot as plt 
import numpy as np
import random
import math

RATIO = 0.8
N = 610
M = 9742
D = 30
lr = 0.005
EPOCH = 100
sigma_0 = 1
sigma_Y = 2
sigma_W = 2
sigma_V = 2

def SGD(train,test,count_matrix_train):
    Y = np.random.normal(0, sigma_0, (N, D))
    W = np.random.normal(0, sigma_0, (M, D))
    V = np.random.normal(0, sigma_0, (M, D))
    U = Y
    rmse=[]
    loss=[]
    rms=RMSE(U,V,train)
    rmse.append(rms)
    print(rmse)
    I= [ [ 0 for i in range(M) ] for j in range(N) ]
    for i,j,_ in train:
        I[i][j]=1
    I=np.array(I)   
    for trial in range(EPOCH):
        print(trial)
        los=0.0
        total_e = 0
        total_Y = 0
        total_V = 0
        total_W = 0
        for i,j,r in train:
            Yi = np.array(Y[i])
            Vj = np.array(V[j])
            Ii = I[i]
            Ui=Yi+np.dot(Ii,W)/sum(Ii)
            U[i] = np.array(Ui)
            predict = sigmoid(np.dot(Ui,Vj))
            e=r-predict
            square_old_yi = sum(np.square(Yi))
            square_old_vj = sum(np.square(Vj))
            Y[i] += lr*(e*predict*(1-predict)*Vj-sigma_Y*Yi)
            np.clip(Y[i], -100000,100000)
            W_value = sigma_W *np.sum(np.sum(np.square(W[count_matrix_train[i]])))/len(count_matrix_train[i])
            W+=lr*(-sigma_W*W+e*predict*(1-predict)/len(count_matrix_train[i])*np.tile(Vj,(M,1)))
            np.clip(W[count_matrix_train[i]], -100000,100000)
            V[j] += lr*(e*predict*(1-predict)*Ui-sigma_V*Vj)
            np.clip(V[i], -100000,100000)
            los+=0.5*(e**2+sigma_Y*square_old_yi+sigma_V*square_old_vj+W_value)
            total_e += abs(e)
            total_Y += square_old_yi
            total_V += square_old_vj
            total_W += W_value
        loss.append(los/len(train))
        print(f"total_e: {total_e}, total_Y: {total_Y}, total_V:{total_V}, total_W:{total_W}")
        rms=RMSE(U,V,test)
        rmse.append(rms)
        print(rms)
    return loss,rmse



def sigmoid(value):
    if value > 20:
        value = 20
    if value < -20:
        value = -20
    return 1/(1+math.exp(-value))


def RMSE(U,V, test):
    count=len(test)
    sum_rmse=0.0
    for i,j,real in test:
        predict = sigmoid(np.dot(U[i],V[i]))
        sum_rmse+=np.square(real-predict)
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

def count_matrix_generation(list_):
    count_matrix = [[] for _ in range(N)]
    for user,movie, _ in list_:
        count_matrix[user].append(movie)
    return count_matrix

def count_movie_generation(list_):
    movie = [0 for _ in range(M)]
    for user,movie_id, _ in list_:
        movie[movie_id]+=1
    return movie

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
    print(len(train))
    count_train = count_matrix_generation(train)
    count_movie = count_movie_generation(train)
    loss,rmse =SGD(train,test,count_train, count_movie)
    plotting(loss, 'loss')
    plotting(rmse, 'rmse')
    
         
if __name__ == '__main__': 
    main()