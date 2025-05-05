import numpy as np
from numpy.linalg import inv as inv
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
import matplotlib.pyplot as plt

RATIO = 0.8
N = 610
M = 9742
D = 30

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]
 
def compute_rmse(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
 
def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

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


def sample_factor_U(
    alpha_sparse_mat, 
    alpha_ind, 
    U, 
    V, 
    alpha, 
    beta0 = 1):
    """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""
    '''
    alpha_sparse_mat,  观测矩阵*alpha
    alpha_ind, 示性矩阵*alpha
    U,  路段特征矩阵 [dim1,rank]
    V,  时间特征矩阵 [dim2,rank]
    alpha, #WX和Y之间的精度（协方差矩阵的倒数）
    beta0 = 1, 
    '''
    dim1, rank = U.shape
    U_bar = np.mean(U, axis = 0)
    #每一列元素取一个均值，一共 rank个元素，U_bar是U矩阵各个行向量的均值向量
 
    temp = dim1 / (dim1 + beta0)
    #N/N+β0
 
    var_mu_hyper = temp * U_bar
    #N U_bar/N+β0，由于μ0为0，所以这个也可以看成μ0*
 
    var_U_hyper = inv(np.eye(rank) 
                      + cov_mat(U, U_bar) 
                      + temp * beta0 * np.outer(U_bar, U_bar))
    #np.eye(rank) 就是I，即我们的W0
    # cov_mat(U, U_bar) 就是N S_bar
    #np.outer(U_bar, U_bar) 相当于 U_bar.reshape(dim1,1) @ U_bar.reshape(1,dim)
    #temp * beta0 * np.outer(U_bar, U_bar)) 即求W0*的最后一项
    #所以这个整体可以看成W0*
 
    var_Lambda_hyper = wishart.rvs(df = dim1 + rank, scale = var_U_hyper)
    #df就是v0*
    #这一部分是从威沙特分布里面取一个[rank,rank]的矩阵样本
    #也就是 Lambda_U
 
    var_mu_hyper = np.random.multivariate_normal(
        var_mu_hyper, 
        np.linalg.inv((dim1 + beta0) * var_Lambda_hyper))
    #从N(μ0*，(β0* Lambda_U)……-1)中选取一个μU向量
    #[rank,1]
 
################################################
################################################
##   上面的部分是在采样ΘU，下面的部分在采样U     ##
################################################
################################################
    
    var1 = V.T
    #[rank,dim2]
 
    var2 = kr_prod(var1, var1)
    #[ranl*rank,dim2]
    #alpha_ind [dim1,dim2]
 
    var3 = (var2 @ alpha_ind.T).reshape([rank, rank, dim1])+ var_Lambda_hyper[:, :, np.newaxis]
    #第一项相当于有值的部分相乘
    #var_Lambda_hyper——[rank,rank],这里给他强行添加了一个维度
    #第一项是12式的右边，第二项是12式的左边
 
    var4 = var1 @ alpha_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    #第一项是13式的第一项，第二项是13式的第二项 【不含最前面的-1】
 
    for i in range(dim1):
        U[i, :] = np.random.multivariate_normal(
                np.linalg.inv(var3[:, :, i]) @ var4[:, i], 
                var3[:, :, i])
    #采样，每一行是rank维度的一个多元正态分布向量
 
    
    return U



def sample_factor_V(
    alpha_sparse_mat, 
    alpha_ind, 
    U, 
    V, 
    beta0 = 1):
    
    dim2, rank = V.shape
    V_bar = np.mean(V, axis = 0)
    #每一列元素去一个均值，一共 rank个元素
    temp = dim2 / (dim2 + beta0)
    #M/M+β0
    var_mu_hyper = temp * V_bar
    #M X_bar/M+β0，由于μ0为0，所以这个也可以看成μ0*
    var_V_hyper = inv(np.eye(rank) 
                      + cov_mat(V, V) 
                      + temp * beta0 * np.outer(V_bar, V_bar))
    #np.eye(rank) 就是I，即我们的X0
    # cov_mat(V, V_bar) 就是N S_bar
    #np.outer(V_bar, V_bar) 相当于 V.reshape(dim2,1) @ V_bar.reshape(1,dim2)
    #temp * beta0 * np.outer(V_bar, V_bar)) 即求X0*的最后一项
    #所以这个可以看成X0*
    var_Lambda_hyper = wishart.rvs(df = dim2 + rank, scale = var_V_hyper)
    #df就是v0*
    #这一部分是从威沙特分布里面去一个[rank,rank]的样本
    #也就是 Lambda_V
    var_mu_hyper = np.random.multivariate_normal(
        var_mu_hyper, 
        (dim2 + beta0) * var_Lambda_hyper)
    #从N(μ0*，(β0* Lambda_X)……-1)中选取一个μV
    #直接multivariate_normal即可
    #[rank,1]
    var1 = U.T
    #[rank,dim1]
    var2 = kr_prod(var1, var1)
    #[ranl*rank,dim2]
    #alpha_ind [dim1,dim2]
    var3 = (var2 @ alpha_ind).reshape([rank, rank, dim2])  + var_Lambda_hyper[:, :, np.newaxis]
    #第一项相当于有值的部分相乘
    #var_Lambda_hyper——[rank,rank],这里给他强行添加了一个维度
    #第一项是12式的右边，第二项是12式的左边
    var4 = var1 @ alpha_sparse_mat + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    #采样，每一行是一个多元正态分布
    for t in range(dim2):
        V[t, :] = np.random.multivariate_normal(
            np.linalg.inv(var3[:, :, t]) @ var4[:, t], 
            var3[:, :, t])
 
    return V

def BPMF(dense_mat, sparse_mat, init, rank, burn_iter, gibbs_iter):
    """Bayesian Probabilistic Matrix Factorization, BPMF."""
    #dense_mat, ————没有丢失值得矩阵
    #sparse_mat, ——有了丢失值得矩阵
    #init, ——初始的W和X矩阵
    #rank, ——特征矩阵的秩
    #burn_iter, ——表示经过了这么多步之后，MCMC代表的马氏链趋近于平稳状态
    #gibbs_iter——吉布斯采样的步数
    
    dim1, dim2 = sparse_mat.shape
    #(214, 8784)
 
    U = init["U"]
    V = init["V"]
 
 
    ind = sparse_mat != 0
    #相当于ind = （sparse_mat != 0）
 
    pos_obs = np.where(ind)
    #输出满足条件 (即这个位置路段的速度非0) 元素的坐标。
 
    pos_test = np.where((dense_mat != 0) & (sparse_mat == 0))
    #在没有丢失数据的矩阵上有值，但是我们人为丢掉的那些点的坐标
 
    dense_test = dense_mat[pos_test]
    #测试集，在没有丢失数据的矩阵上有值，但是我们人为丢掉的那些点的坐标
 
    alpha = 2 
    #WX和Y之间的精度（协方差矩阵的倒数）
 
    U_plus = np.zeros((dim1, rank))
    V_plus = np.zeros((dim2, rank))
    #多次采样的U和V的和
 
    temp_hat = np.zeros(sparse_mat.shape)
    #多次采样的预测矩阵R的和（在平稳状态之前）
 
    show_iter = 200
    mat_hat_plus = np.zeros(sparse_mat.shape)
    #多次采样的预测矩阵R的和（在平稳状态之后）
 
    for it in range(burn_iter + gibbs_iter):
        alpha_ind = alpha * ind #有观测值的哪些点*alpha
        alpha_sparse_mat = alpha * sparse_mat #观测矩阵*alpha
        U = sample_factor_U(alpha_sparse_mat, alpha_ind, U, V, alpha)
        #采样一个W矩阵
        V = sample_factor_V(alpha_sparse_mat, alpha_ind, U, V)
        #采样一个X矩阵
        mat_hat = U @ V.T
        temp_hat += mat_hat
        if (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat = temp_hat / show_iter
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(dense_test, temp_hat[pos_test])))
            print('RMSE: {:.6}'.format(compute_rmse(dense_test, temp_hat[pos_test])))
            temp_hat = np.zeros(sparse_mat.shape)
            print()
        if it + 1 > burn_iter:
            U_plus += U
            V_plus += V
            mat_hat_plus += mat_hat
    mat_hat = mat_hat_plus / gibbs_iter
    U = U_plus / gibbs_iter
    V = V_plus / gibbs_iter
    print('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, mat_hat[pos_test])))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, mat_hat[pos_test])))
    print()
    
    return mat_hat, U, V

def main():
    movie_dir="movies.csv"
    rating_dir = "ratings.csv"
    mapping_id = read_id(movie_dir)
    train,test=Load_data(rating_dir,mapping_id)
    train = r_generation(train)
    test = r_generation(test)
    init = {
    "U": 0.01 * np.random.randn(N, D), 
    "V": 0.01 * np.random.randn(M, D)}
    mat_hat, U, V = BPMF(test, train, init, 30, 1, 200)
if __name__ == '__main__': 
    main()