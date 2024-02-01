import numpy as np

import pickle

n = 30

ev = np.random.exponential(5,n)

Q = 0

for i in range(n):
    
    v = np.random.normal(0,1,n)
    v = v/np.linalg.norm(v)
    
    Q = Q + ev[i] * np.outer(v,v)


pickle.dump(Q, open('./raw_data/Q', 'wb'))


def func_large(x):
    
    return ( np.dot(x, np.matmul(Q,x) ))*(3./4)

def func_small(x):
    
    return ( np.dot(x, np.matmul(Q,x) ))*(1./4)


def sample_stiefel(n,k): 
    
    U = np.random.normal( 0, 1, (n, k) ) 
    
    L,S,R = np.linalg.svd(np.matmul(U.T, U)) 
    
    U_ = np.matmul( np.matmul(L, np.diag( S**(-1./2) ) ), R ) 
        
    return np.matmul(U,U_) 


def get_grad_est(func,x,n,k, delta):
    
    V = sample_stiefel(n,k) 
    
    res = 0
    
    for i in range(k):
        
        res = res + ( func (x + delta*V[:,i]) - func (x - delta*V[:,i]) ) * V[:,i] / 2
        
    res = res * n / k / delta
    
    return res


# ITER = 10000
# k = 80

def get_res(k, eta, ITER = 10000, large=True, rep = 10 ):

    res_overall = []
    
    for _ in range(rep): 
        
        x = np.random.normal(0,1,d)

        delta = 0.1

        x_norms = [np.linalg.norm(x)]

        ys = [func(x)]

        for i in range(ITER): 

            x = x - eta * get_grad_est(func,x,n,k,delta)

            x_norms.append(np.linalg.norm(x)) 
            ys.append(func(x)) 

            delta = np.max([ 0.0001, delta/2.]) 
            
        res_overall.append((x_norms, ys))
        
    if large:
        pickle.dump( res_overall, open('./raw_data/res_k{0}_eta{1}_explarge'.format(k,eta),'wb')) 
    else:
        pickle.dump( res_overall, open('./raw_data/res_k{0}_eta{1}_expsmall'.format(k,eta),'wb')) 


def get_res(k,eta,ITER = 10000, large=True, rep = 10 ): 

    res_overall = [] 
    
    for _ in range(rep): 
        
        x = np.random.normal(0,1,n) 

        delta = 0.1 

        x_norms = [np.linalg.norm(x)] 
        
        if large: 

            ys = [func_large(x)] 
            
        else: 
            
            ys = [func_small(x)] 

        for i in range(ITER): 
            
            if large:
                x = x - eta * get_grad_est(func_large,x,n,k,delta) 

                x_norms.append(np.linalg.norm(x)) 
                ys.append(func_large(x)) 
            else:
                x = x - eta * get_grad_est(func_small,x,n,k,delta) 

                x_norms.append(np.linalg.norm(x)) 
                ys.append(func_small(x)) 

            delta = np.max([ 0.0001, delta/2.]) 
            
        res_overall.append((x_norms, ys)) 
        
    if large:
        pickle.dump( res_overall, open('./raw_data/res_k{0}_eta{1}_explarge'.format(k,eta),'wb')) 
    else:
        pickle.dump( res_overall, open('./raw_data/res_k{0}_eta{1}_expsmall'.format(k,eta),'wb')) 


def get_res_grad(eta,ITER = 10000, large=True, rep = 10 ):

    res_overall = []
    
    for _ in range(rep): 
        
        x = np.random.normal(0,1,n)

        delta = 0.1

        x_norms = [np.linalg.norm(x)]
        
        if large: 

            ys = [func_large(x)] 
            
        else: 
            
            ys = [func_small(x)] 

        for i in range(ITER): 
            
            if large:
                tmp = np.matmul(Q,x)
                grad = 3/2 * (np.dot( tmp , x ))**(-1./4) * tmp
            else:
                tmp = np.matmul(Q,x)
                grad = 1/2 * (np.dot( tmp , x ))**(-3./4) * tmp

            x = x - eta * grad 

            x_norms.append(np.linalg.norm(x)) 
            
            if large:
                ys.append(func_large(x)) 
            else:
                ys.append(func_small(x)) 

            delta = np.max([ 0.00001, delta/2.]) 
            
        res_overall.append((x_norms, ys)) 
        
    if large:
        pickle.dump( res_overall, open('./raw_data/res_gd_eta{}_explarge'.format(eta),'wb')) 
    else:
        pickle.dump( res_overall, open('./raw_data/res_gd_eta{}_expsmall'.format(eta),'wb')) 

        