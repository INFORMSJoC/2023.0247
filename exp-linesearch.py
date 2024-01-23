import numpy as np

import pickle

n = 30

ev = np.random.exponential(5,n)

Q = 0

for i in range(n):
    
    v = np.random.normal(0,1,n)
    v = v/np.linalg.norm(v)
    
    Q = Q + ev[i] * np.outer(v,v)


# Q = pickle.load(open('./raw_data/Q','rb'))


def func_large(x):
    
    return ( np.dot(x, np.matmul(Q,x) ))*(3./4)

def func_small(x):
    
    return ( np.dot(x, np.matmul(Q,x) ))*(1./4)


def sample_stiefel(n,k): 
    
    U = np.random.normal( 0, 1, (n, k) ) 
    
    L,S,R = np.linalg.svd(np.matmul(U.T, U)) 
    
    U_ = np.matmul( np.matmul(L, np.diag( S**(-1./2) ) ), R ) 
        
    return np.matmul(U,U_) 


def get_grad_est(x,n,k, delta,large):
    
    if large:
        func = func_large
    else:
        func = func_small
    
    V = sample_stiefel(n,k) 
    
    res = 0
    
    for i in range(k):
        
        res = res + ( func (x + delta*V[:,i]) - func (x - delta*V[:,i]) ) * V[:,i] / 2
        
    res = res * n / k / delta
    
    return res

def linsearch(x,eta,grad,eps,large):
    
    if large:
        func = func_large
    else:
        func = func_small
    
    z_minus = x
    
    z_plus = x - eta * grad 
    
    while (np.linalg.norm(z_plus - z_minus) > eps):
        
#         print(np.linalg.norm(z_plus - z_minus))
    
        z_mid = (z_minus + z_plus) / 2.
        
        y_minus = func(z_minus); y_plus = func(z_plus); y_mid = func(z_mid); 
        
        if (max([y_minus,y_plus,y_mid]) == y_minus):
            
            z_minus = z_mid
            
        elif (max([y_minus,y_plus,y_mid]) == y_plus):
            
            z_plus = z_mid
            
        else: 
            break
            
    return z_plus

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
            
            grad = get_grad_est(x,n,k,delta,large) 
            x = linsearch(x,0.01,grad,0.0001,large) 
#             x = x - eta * get_grad_est(x,n,k,delta) 

            x_norms.append(np.linalg.norm(x)) 
            
            if large:
                ys.append(func_large(x)) 
            else:
                ys.append(func_small(x)) 

            delta = np.max([ 0.0001, delta/2.]) 
            
        res_overall.append((x_norms, ys)) 
        
    if large:
        pickle.dump( res_overall, open('./raw_data/newres_k{0}_eta{1}_explarge'.format(k,eta),'wb')) 
    else:
        pickle.dump( res_overall, open('./raw_data/newres_k{0}_eta{1}_expsmall'.format(k,eta),'wb'))  