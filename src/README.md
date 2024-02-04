
# Additional details 

## The file "exp.py"

- the function func_large is F_1 defined in the paper.  
- the function func_small is F_2 defined in the paper.  
- the function sample_stiefel returns a random k-frame.  

  -- parameters: 
    ---n: ambient dimension. k = 30 for all experiments in the paper. 
    ---k: number of random directions. should be one of {1,10,20,30}. 

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


def get_grad_est(func,x,n,k, delta):
    
    V = sample_stiefel(n,k) 
    
    res = 0
    
    for i in range(k):
        
        res = res + ( func (x + delta*V[:,i]) - func (x - delta*V[:,i]) ) * V[:,i] / 2
        
    res = res * n / k / delta
    
    return res


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
        pickle.dump( res_overall, open('../raw_data/res_k{0}_eta{1}_explarge'.format(k,eta),'wb')) 
    else:
        pickle.dump( res_overall, open('../raw_data/res_k{0}_eta{1}_expsmall'.format(k,eta),'wb')) 


# the function get_res_gred returns results for GD on a Lojasiewicz function in log scale. 

# parameters: 
## eta: learning rate, learning rate is set to 0.005 for all experiments in the paper. 
## ITER: number of total iterations. 
## large: when set to True, the objective function is F_1 defined in the paper. 
##        when set to False, the objective function is F_2 defined in the paper. 
## rep: repeat the experiments for "rep" number of times. By default, rep is set to 10. 

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
        pickle.dump( res_overall, open('../raw_data/res_gd_eta{}_explarge'.format(eta),'wb')) 
    else:
        pickle.dump( res_overall, open('../raw_data/res_gd_eta{}_expsmall'.format(eta),'wb')) 

    
if __name__ == '__main__':

    n = 30

    eta = 0.005

    ev = np.random.exponential(5,n)

    Q = 0

    for i in range(n):
        
        v = np.random.normal(0,1,n)
        v = v/np.linalg.norm(v)
        
        Q = Q + ev[i] * np.outer(v,v)

    pickle.dump(Q, open('../raw_data/Q', 'wb'))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--plain_gd', type = int )
    parser.add_argument('--k', type = int )
    parser.add_argument('--large', type = int )
    args = parser.parse_args() 

    plain_gd = bool(args.plain_gd)

    k = args.k

    large = bool(args.large)


    if (plain_gd == 1):

        get_res_grad( eta, ITER = 15000, large = large) 

    else:

        get_res(k, eta, ITER = 15000, large = large) 


