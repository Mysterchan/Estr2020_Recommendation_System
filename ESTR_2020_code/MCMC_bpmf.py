import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import wishart
import matplotlib.pyplot as plt 
import numpy as np
import math

RATIO = 0.8
N = 610
M = 9742
D = 60
EPOCH = 1000

def sampling_normal_wishart(wishart_matrix, u_mean_parameter, v_freedom_parameter, beta_parameter):
    Lambda = wishart(df=v_freedom_parameter, scale=wishart_matrix).rvs() 
    cov = np.linalg.inv(beta_parameter * Lambda)
    normal_vector = multivariate_normal(u_mean_parameter, cov)
    return normal_vector, Lambda, cov

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

def r_generation(list_):
    R = np.full((N,M),-1)
    for user_id, movie_id, rating in list_:
        R[user_id, movie_id] = rating
    return R    

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

def sigmoid(value):
    if value > 20:
        value = 20
    return (math.e**value)/(1+math.e**value)

def BPMF(U_in, V_in, Step, R, R_test, length_train, length_test):
    old_U = np.array(U_in)
    old_V = np.array(V_in)
    ans = np.zeros((N, M))
    loss_list = []
    rmse_list = []

    # initialize now the hierarchical priors:
    alpha = 2  # observation noise, they put it = 2 in the paper
    mean_hyper_0_V = np.zeros(D)
    mean_hyper_0_U = np.zeros(D)
    v_freedom_U = D
    v_freedom_V = D
    Beta_0_U = 2
    Beta_0_V = 2
    W_0_U = np.eye(D)
    W_0_V = np.eye(D)
    
    for trial in range(1,Step+1):

        '''Here it will try to compute what the author suggested for Gaussian-Wishart distributio to  sample the hyperparameter'''
        # pre_processing on the hyperparamter on movie:
        Beta_0_V_star = Beta_0_V + M
        v_freedom_V_star = v_freedom_V + M
        W_0_V_inv = np.linalg.inv(W_0_V)  

        V_average = np.mean(old_V, axis = 0)
        mean_hyper_0_star_V = (Beta_0_V * mean_hyper_0_V + M * V_average) / (Beta_0_V_star)
        S_V = np.dot(old_V.T, old_V)/M
        W_0_star_V_inv = W_0_V_inv + M * S_V + Beta_0_V * M / (Beta_0_V_star) * np.outer(mean_hyper_0_star_V - V_average, mean_hyper_0_star_V - V_average)
        W_0_star_V = np.linalg.inv(W_0_star_V_inv)
        mu_V, Lambda_V, _ = sampling_normal_wishart(W_0_star_V,mean_hyper_0_star_V, v_freedom_V_star,Beta_0_V_star)

        # pre_processing on the hyperparamter on user:
        Beta_0_U_star = Beta_0_U + N
        v_freedom_U_star = v_freedom_U + N
        W_0_U_inv = np.linalg.inv(W_0_U)

        U_average = np.mean(old_U, axis = 0)
        mean_hyper_0_star_U = (Beta_0_U * mean_hyper_0_U + N * U_average) / Beta_0_U_star
        S_U = np.dot(old_U.T, old_U)/N
        W_0_star_U_inv = W_0_U_inv + N * S_U + Beta_0_U * N / (Beta_0_U_star) *np.outer(mean_hyper_0_star_U - U_average, mean_hyper_0_star_U - U_average)
        W_0_star_U = np.linalg.inv(W_0_star_U_inv)
        mu_U, Lambda_U, _ = sampling_normal_wishart(W_0_star_U,mean_hyper_0_star_U, v_freedom_U_star,Beta_0_U_star)

        """Here will be the sampling by the normal distribution suggested by the author"""
        """ For each i = 1, ..., N sample user features"""
        U_new = np.zeros((N,D))
        V_new = np.zeros((M,D))

        for i in range(N):  
            Lambda_U_2 = np.zeros((D, D)) 
            mu_i_star_1 = np.zeros(D) 
            for j in range(M):
                if R[i,j]!=-1:
                    Lambda_U_2 +=  np.outer(old_V[j], old_V[j])
                    mu_i_star_1 += old_V[j] * R[i, j] 

            Lambda_i_star_U = Lambda_U + alpha * Lambda_U_2
            Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_U,mu_U) 
            mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
            U_new[i] = multivariate_normal(mu_i_star, Lambda_i_star_U_inv)
        old_U = np.array(U_new)

        """ For each i = 1, ..., M sample movie feature"""
        for i in range(M):
            Lambda_V_2 = np.zeros((D, D)) 
            mu_i_star_1 = np.zeros(D)
            for j in range(N): 
                if R[j,i]!=-1:
                    Lambda_V_2 += np.outer(old_U[j], old_U[j])
                    mu_i_star_1 += old_U[j] * R[j, i]  

            Lambda_j_star_V = Lambda_V + alpha * Lambda_V_2
            Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_V, mu_V)
            mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            V_new[i] = multivariate_normal(mu_j_star, Lambda_j_star_V_inv)
        old_V = np.array(V_new)
        
        R_step = np.dot(U_new,V_new.T)
        for i in range(N):  # reduce all the predictions to the correct ratings range.
            for j in range(M):
                ans[i,j] = max((ans[i,j]*trial  + R_step[i,j])/trial,0)
                ans[i,j] = min(ans[i,j],1)

        loss = rmse = 0
        for i in range(N):
            for j in range(M):
                if R[i, j] != -1: 
                    loss += (ans[i, j] - R[i, j])**2
                if R_test[i, j] != -1:
                    rmse += (ans[i, j] - R_test[i, j]) ** 2

        # Append sqrt of average loss and RMSE to their respective lists
        loss_list.append(math.sqrt(loss / length_train))
        rmse_list.append(math.sqrt(rmse / length_test))
    
    return loss_list, rmse_list

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
    length_t = len(train)
    length_test = len(test)
    train = r_generation(train)
    test = r_generation(test)
    new_U = np.zeros((N,D))
    new_V = np.zeros((M,D))
    loss,rmse = BPMF(new_U, new_V,EPOCH, train, test, length_t, length_test)
    print(loss, rmse)
    plotting(loss, 'loss')
    plotting(rmse, 'rmse')
         
if __name__ == '__main__': 
    main()