import numpy as np

import pickle

import argparse

# load the matrix Q as a global variable 
Q = pickle.load(open('../raw_data/Q','rb'))


# the function func_large is F_1 defined in the paper.  

def func_large(x):
    
    return ( np.dot(x, np.matmul(Q,x) ))*(3./4)


# the function func_small is F_2 defined in the paper.  

def func_small(x):
    
    return ( np.dot(x, np.matmul(Q,x) ))*(1./4)

# the function sample_stiefel returns a random k-frame.  

# parameters: 
## n: ambient dimension. k = 30 for all experiments in the paper. 
## k: number of random directions. should be one of {1,10,20,30}. 

def sample_stiefel(n,k): 
    
    U = np.random.normal( 0, 1, (n, k) ) 
    
    L,S,R = np.linalg.svd(np.matmul(U.T, U)) 
    
    U_ = np.matmul( np.matmul(L, np.diag( S**(-1./2) ) ), R ) 
        
    return np.matmul(U,U_) 


# the function get_grad_est returns a gradient estimator.  

# parameters: 
## func: the objective functions. 
## n: ambient dimension. k = 30 for all experiments in the paper. 
## k: number of random directions. should be one of {1,10,20,30}. 
## delta: finite difference granularity. 

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


# the function linesearch returns the next iterate point using binary linesearch. 

# parameters: 
## x: the current iterate. 
## eta: learning rate, learning rate is set to 0.005 for all experiments in the paper. 
## grad: the direction along which linesearch is performed. 
## eps: linesearch tolerance. If the distance between two consecutive points between the binary linesearch procedure
##      is less then eps, then terminate. 
## large: when set to True, the objective function is F_1 defined in the paper. 
##        when set to False, the objective function is F_2 defined in the paper. 

def linsearch(x,eta,grad,eps,large):
    
    if large:
        func = func_large
    else:
        func = func_small
    
    z_minus = x
    
    z_plus = x - eta * grad 
    
    while (np.linalg.norm(z_plus - z_minus) > eps):
    
        z_mid = (z_minus + z_plus) / 2.
        
        y_minus = func(z_minus); y_plus = func(z_plus); y_mid = func(z_mid); 
        
        if (max([y_minus,y_plus,y_mid]) == y_minus):
            
            z_minus = z_mid
            
        elif (max([y_minus,y_plus,y_mid]) == y_plus):
            
            z_plus = z_mid
            
        else: 
            break
            
    return z_plus



# the function get_res returns results for ZGD on a Lojasiewicz function in log scale. 

# parameters: 
## k: number of random directions. should be one of {1,10,20,30}. 
## eta: learning rate, learning rate is set to 0.005 for all experiments in the paper. 
## ITER: number of total iterations. 
## large: when set to True, the objective function is F_1 defined in the paper. 
##        when set to False, the objective function is F_2 defined in the paper. 
## rep: repeat the experiments for "rep" number of times. By default, rep is set to 10. 

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
            x = linsearch(x,0.5,grad,0.0001,large) 

            x_norms.append(np.linalg.norm(x)) 
            
            if large:
                ys.append(func_large(x)) 
            else:
                ys.append(func_small(x)) 

            delta = np.max([ 0.0001, delta/2.]) 
            
        res_overall.append((x_norms, ys)) 
        
    if large:
        pickle.dump( res_overall, open('../raw_data/newres_k{0}_eta{1}_explarge'.format(k,eta),'wb')) 
    else:
        pickle.dump( res_overall, open('../raw_data/newres_k{0}_eta{1}_expsmall'.format(k,eta),'wb'))  


if __name__ == '__main__':

    n = 30

    eta = 0.005

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--plain_gd', type = int )
    parser.add_argument('--k', type = int )
    parser.add_argument('--large', type = int )
    args = parser.parse_args() 

    # plain_gd = bool(args.plain_gd)

    k = args.k

    large = bool(args.large)

    get_res(k, eta, ITER = 12000, large = large) 

