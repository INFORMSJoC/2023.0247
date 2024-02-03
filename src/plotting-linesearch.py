#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 15 


# plot the linesearch results in log scale. 

# parameters: 
## eta: learning rate, all learning rate is set to 0.005
## k: number of random directions. should be one of {1,10,20,30}. 
## f: When set to True, the y-axis is set to function value f(x). 
##    When set to False, the y-axis is set to \| x - x^* \|, which is \| \x \| for the experiments in the paper. 
## small: when set to True, the objective function is F_2 defined in the paper. 
##        when set to False, the objective function is F_1 defined in the paper. 

def plot_iter_log(eta ,k, f = True, small = True): 
        
    if small: 
        res = pickle.load(open('../raw_data/newres_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
    else: 
        res = pickle.load(open('../raw_data/newres_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 

    if f: 
        m = np.mean( np.array( res )[:,1,:] , axis = 0 ) 
        std = np.std( np.array( res )[:,1,:] , axis = 0 ) 
        plt.plot( m , label = 'with line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 

    else: 
        m = np.mean( np.array( res )[:,0,:] , axis = 0 ) 
        std = np.std( np.array( res )[:,0,:] , axis = 0 ) 
        plt.plot( m , label = 'with line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 

    if small: 
        res = pickle.load(open('../raw_data/res_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
    else: 
        res = pickle.load(open('../raw_data/res_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 

    if f: 
        m = np.mean( np.array( res )[:,1,:] , axis = 0 ) 
        std = np.std( np.array( res )[:,1,:] , axis = 0 ) 
        plt.plot( m , label = 'without line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 

    else: 
        m = np.mean( np.array( res )[:,0,:] , axis = 0 )
        std = np.std( np.array( res )[:,0,:] , axis = 0 ) 
        plt.plot( m , label = 'without line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 
        
    plt.legend() 

    plt.xlabel('Iterations') 
    
    plt.gca().set_yscale('log')
    
    if f: 
        plt.ylabel(r'$f(x)$') 
        plt.xlim([-50, 3000]) 
        plt.ylim([-3, 50]) 
        
    else: 
        plt.ylabel(r'$\| x \|$') 
        plt.xlim([-100, 10000]) 
        plt.ylim([-1, 6.5]) 
        
    plt.rcParams['figure.figsize'] = 6,4

    
    if (small and f): 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_f_k{}_expsmall_iter_log.png'.format(k) , dpi = 150)
    elif (small and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_x_k{}_expsmall_iter_log.png'.format(k) , dpi = 150)
    elif ((not small) and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_x_k{}_explarge_iter_log.png'.format(k) , dpi = 150)
    else: 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_f_k{}_explarge_iter_log.png'.format(k) , dpi = 150)
        
    plt.clf() 


plot_iter_log(0.005 , 1, f = True, small = True) 
plot_iter_log(0.005 , 1, f = False, small = True) 
plot_iter_log(0.005 , 1, f = True, small = False) 
plot_iter_log(0.005 , 1, f = False, small = False) 

plot_iter_log(0.005 , 10, f = True, small = True) 
plot_iter_log(0.005 , 10, f = False, small = True) 
plot_iter_log(0.005 , 10, f = True, small = False) 
plot_iter_log(0.005 , 10, f = False, small = False) 

plot_iter_log(0.005 , 20, f = True, small = True) 
plot_iter_log(0.005 , 20, f = False, small = True) 
plot_iter_log(0.005 , 20, f = True, small = False) 
plot_iter_log(0.005 , 20, f = False, small = False) 

plot_iter_log(0.005 , 30, f = True, small = True) 
plot_iter_log(0.005 , 30, f = False, small = True) 
plot_iter_log(0.005 , 30, f = True, small = False) 
plot_iter_log(0.005 , 30, f = False, small = False) 


plt.rcParams['font.size'] = 15 

# plot the linesearch results in linear scale

# parameters: 
## eta: learning rate, all learning rate is set to 0.005
## k: number of random directions. should be one of {1,10,20,30}. 
## f: When set to True, the y-axis is set to function value f(x). 
##    When set to False, the y-axis is set to \| x - x^* \|, which is \| \x \| for the experiments in the paper. 
## small: when set to True, the objective function is F_2 defined in the paper. 
##        when set to False, the objective function is F_1 defined in the paper. 

def plot_iter(eta ,k, f = True, small = True): 
        
    if small: 
        res = pickle.load(open('../raw_data/newres_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
    else: 
        res = pickle.load(open('../raw_data/newres_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 

    if f: 
        m = np.mean( np.array( res )[:,1,:] , axis = 0 ) 
        std = np.std( np.array( res )[:,1,:] , axis = 0 ) 
        plt.plot( m , label = 'with line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 

    else: 
        m = np.mean( np.array( res )[:,0,:] , axis = 0 ) 
        std = np.std( np.array( res )[:,0,:] , axis = 0 ) 
        plt.plot( m , label = 'with line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 

    if small: 
        res = pickle.load(open('../raw_data/res_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
    else: 
        res = pickle.load(open('../raw_data/res_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 

    if f: 
        m = np.mean( np.array( res )[:,1,:] , axis = 0 ) 
        std = np.std( np.array( res )[:,1,:] , axis = 0 ) 
        plt.plot( m , label = 'without line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 

    else: 
        m = np.mean( np.array( res )[:,0,:] , axis = 0 )
        std = np.std( np.array( res )[:,0,:] , axis = 0 ) 
        plt.plot( m , label = 'without line search' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 
        
    plt.legend() 

    plt.xlabel('Iterations') 
    
    plt.gca().set_yscale('log')
    
    if f: 
        plt.ylabel(r'$f(x)$') 
        plt.xlim([-50, 3000]) 
        plt.ylim([-3, 50]) 
        
    else: 
        plt.ylabel(r'$\| x \|$') 
        plt.xlim([-100, 10000]) 
        plt.ylim([-1, 6.5]) 
        
    plt.rcParams['figure.figsize'] = 6,4
    
    if (small and f): 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_f_k{}_expsmall_iter.png'.format(k) , dpi = 150)
    elif (small and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_x_k{}_expsmall_iter.png'.format(k) , dpi = 150)
    elif ((not small) and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_x_k{}_explarge_iter.png'.format(k) , dpi = 150)
    else: 
        plt.tight_layout() 
        plt.savefig( '../figures/lsfig_f_k{}_explarge_iter.png'.format(k) , dpi = 150)
        
    plt.clf() 




plot_iter(0.005 , 1, f = True, small = True) 
plot_iter(0.005 , 1, f = False, small = True) 
plot_iter(0.005 , 1, f = True, small = False) 
plot_iter(0.005 , 1, f = False, small = False) 

plot_iter(0.005 , 10, f = True, small = True) 
plot_iter(0.005 , 10, f = False, small = True) 
plot_iter(0.005 , 10, f = True, small = False) 
plot_iter(0.005 , 10, f = False, small = False) 

plot_iter(0.005 , 20, f = True, small = True) 
plot_iter(0.005 , 20, f = False, small = True) 
plot_iter(0.005 , 20, f = True, small = False) 
plot_iter(0.005 , 20, f = False, small = False) 

plot_iter(0.005 , 30, f = True, small = True) 
plot_iter(0.005 , 30, f = False, small = True) 
plot_iter(0.005 , 30, f = True, small = False) 
plot_iter(0.005 , 30, f = False, small = False) 

