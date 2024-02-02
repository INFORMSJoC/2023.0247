#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import pickle

import matplotlib.pyplot as plt 
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


thetas = np.arange(0.50001,1,0.0001) 

# xs = 

rate_y = 1 / (1-2*thetas) 

rate_x = (1-thetas) / (1-2*thetas) 

gap = thetas / (1-2*thetas) 


# In[3]:


plt.rcParams['font.size'] = 15 

l1, = plt.plot(thetas,rate_y, label = r'$\frac{1}{1 - 2\theta}$') 

l2, = plt.plot(thetas,rate_x, label = r'$\frac{1- \theta}{1 - 2\theta}$') 

plt.legend(handles = [l1,l2]) 

plt.xlabel(r"Value of $\theta$") 

plt.ylim(-30,1) 

plt.tight_layout() 

plt.savefig( '../figures/theta.png', dpi = 150) 


# In[4]:


plt.rcParams['font.size'] = 15

l1, = plt.plot(thetas,rate_y, label = r'$\frac{\theta}{1 - 2\theta}$')

# l2, = plt.plot(thetas,rate_x, label = r'$\frac{1- \theta}{1 - 2\theta}$')

plt.legend(handles = [l1], loc = 'lower right')

plt.xlabel(r"Value of $\theta$")

plt.ylim(-30,1)

plt.tight_layout()

plt.savefig( '../figures/theta2.png', dpi = 150)


# # Eigvalues of Q

# In[5]:


Q = pickle.load(open('../raw_data/Q','rb')) 


# In[6]:


plt.hist( np.linalg.eig(Q)[0] , bins = 30 )


# # the toy function

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-200,200,0.01)
y = np.sqrt(np.abs(x))

rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
      "xtick.major.size" : 5, "ytick.major.size" : 5,}

plt.rcParams['font.size'] = 15 
plt.rcParams['figure.figsize'] = 8,3.5

# with plt.rc_context(rc):
fig, ax = plt.subplots()
ax.plot(x, y) 

# plt.ylim(0,100)

ax.spines['left'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# make arrows
ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
        transform=ax.get_xaxis_transform(), clip_on=False)

# plt.show()

plt.tight_layout() 

plt.savefig( '../figures/toy.png' , dpi = 150)


# # in terms of iterations -log scale

# In[8]:


# res[1] 

plt.rcParams['font.size'] = 15 

def plot_iter_log(eta , f = True, small = True): 
    
    plt.rcParams['figure.figsize'] = 6,4

    for k in [1,10,20,30]: 
        
        if small: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
        else: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 
        
        if f: 
            m = np.mean( np.array( res )[:,1,:] , axis = 0 ) 
            std = np.std( np.array( res )[:,1,:] , axis = 0 ) 
            plt.plot( m , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 
            
        else: 
            m = np.mean( np.array( res )[:,0,:] , axis = 0 )
            std = np.std( np.array( res )[:,0,:] , axis = 0 ) 
            plt.plot( m , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 
            
    if small: 
        res = pickle.load(open('../raw_data/res_gd_eta{}_expsmall'.format(eta),'rb')) 
    else: 
        res = pickle.load(open('../raw_data/res_gd_eta{}_explarge'.format(eta),'rb')) 

    if f: 
        m = np.mean( np.array( res )[:,1,:] , axis = 0 )
        std = np.std( np.array( res )[:,1,:] , axis = 0 )
#             l = 
        plt.plot( m , label = 'GD' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4)

    else: 
        m = np.mean( np.array( res )[:,0,:] , axis = 0 )
        std = np.std( np.array( res )[:,0,:] , axis = 0 )

        plt.plot( m , label = 'GD' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4)
    
    plt.gca().set_yscale('log')
    
    plt.legend() 

    plt.xlabel('Iterations') 
    
    if f: 
        plt.ylabel(r'$f(x)$') 
        plt.xlim([-50, 3000]) 
        plt.ylim([-10, 50]) 
        
    else: 
        plt.ylabel(r'$\| x \|$') 
        plt.xlim([-100, 10000]) 
        plt.ylim([-1, 6.5]) 
        

#     plt.tight_layout() 
    
    if (small and f): 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_f_eta{}_expsmall_iter_log.png'.format(eta) , dpi = 150)
    elif (small and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_x_eta{}_expsmall_iter_log.png'.format(eta) , dpi = 150)
    elif ((not small) and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_x_eta{}_explarge_iter_log.png'.format(eta) , dpi = 150)
    else: 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_f_eta{}_explarge_iter_log.png'.format(eta) , dpi = 150)
        
    plt.clf() 


# In[9]:


plot_iter_log(0.005 , f = True, small = True) 
plot_iter_log(0.005 , f = False, small = True) 
plot_iter_log(0.005 , f = True, small = False) 
plot_iter_log(0.005 , f = False, small = False) 


# # in terms of iterations -linear scale

# In[10]:



plt.rcParams['font.size'] = 15 

def plot_iter(eta , f = True, small = True): 
    
    plt.rcParams['figure.figsize'] = 6,4

    for k in [1,10,20,30]: 
        
        if small: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
        else: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 
        
        if f: 
            m = np.mean( np.array( res )[:,1,:] , axis = 0 ) 
            std = np.std( np.array( res )[:,1,:] , axis = 0 ) 
            plt.plot( m , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 
            
        else: 
            m = np.mean( np.array( res )[:,0,:] , axis = 0 )
            std = np.std( np.array( res )[:,0,:] , axis = 0 ) 
            plt.plot( m , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4) 
            
    if small: 
        res = pickle.load(open('../raw_data/res_gd_eta{}_expsmall'.format(eta),'rb')) 
    else: 
        res = pickle.load(open('../raw_data/res_gd_eta{}_explarge'.format(eta),'rb')) 

    if f: 
        m = np.mean( np.array( res )[:,1,:] , axis = 0 )
        std = np.std( np.array( res )[:,1,:] , axis = 0 )
#             l = 
        plt.plot( m , label = 'GD' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4)

    else: 
        m = np.mean( np.array( res )[:,0,:] , axis = 0 )
        std = np.std( np.array( res )[:,0,:] , axis = 0 )

        plt.plot( m , label = 'GD' ) 
        plt.fill_between(range(len(m)), m - std, m + std, alpha = 0.4)
    
#     plt.gca().set_yscale('log')
    
    plt.legend() 

    plt.xlabel('Iterations') 
    
    if f: 
        plt.ylabel(r'$f(x)$') 
        plt.xlim([-50, 3000]) 
        plt.ylim([-10, 50]) 
        
    else: 
        plt.ylabel(r'$\| x \|$') 
        plt.xlim([-100, 10000]) 
        plt.ylim([-1, 6.5]) 
        

#     plt.tight_layout() 
    
    if (small and f): 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_f_eta{}_expsmall_iter.png'.format(eta) , dpi = 150)
    elif (small and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_x_eta{}_expsmall_iter.png'.format(eta) , dpi = 150)
    elif ((not small) and (not f)): 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_x_eta{}_explarge_iter.png'.format(eta) , dpi = 150)
    else: 
        plt.tight_layout() 
        plt.savefig( '../figures/fig_f_eta{}_explarge_iter.png'.format(eta) , dpi = 150)
        
    plt.clf() 


# In[11]:


plot_iter(0.005 , f = True, small = True) 
plot_iter(0.005 , f = False, small = True) 
plot_iter(0.005 , f = True, small = False) 
plot_iter(0.005 , f = False, small = False) 


# In[12]:


# plot_iter_compare(0.001 , f = True, small = True) 
# plot_iter_compare(0.001 , f = False, small = True) 
# plot_iter_compare(0.001 , f = True, small = False) 
# plot_iter_compare(0.001 , f = False, small = False) 


# # in terms of sample complexity - log scale

# In[13]:


plt.rcParams['font.size'] = 15 

def plot_sample_log(eta , f = True, small = True): 
    
    plt.rcParams['figure.figsize'] = 6,4

    for k in [1, 10, 20, 30]:  
        if small: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
        else: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 
        
        if f: 
            m = np.mean( np.array( res )[:,1,:] , axis = 0 )
            std = np.std( np.array( res )[:,1,:] , axis = 0 )
            m_aug = [  ] ; std_aug = []
            
            if k == 1:
                N = 11000
            else:
                N = 2000
    
            for i in range(N):
                m_aug = m_aug + [m[i]]*k*2
                std_aug = std_aug + [std[i]]*k*2
            
            plt.plot( m_aug , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m_aug)), np.array(m_aug) - np.array(std_aug), np.array(m_aug) + np.array(std_aug), alpha = 0.4  )
            
        else: 
            m = np.mean( np.array( res )[:,0,:] , axis = 0 )
            std = np.std( np.array( res )[:,0,:] , axis = 0 )
            
            m_aug = [  ] ; std_aug = []
            
            if k == 1:
                N = 11000
            else:
                N = 2000
            
            for i in range(N):
                m_aug = m_aug + [m[i]]*k*2
                std_aug = std_aug + [std[i]]*k*2
            
            plt.plot( m_aug , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m_aug)), np.array(m_aug) - np.array(std_aug), np.array(m_aug) + np.array(std_aug), alpha = 0.4 ) 

    plt.legend() 
    
    plt.gca().set_yscale('log')

    plt.xlabel('Number of Function Evaluations') 
    
    if f: 
        plt.ylabel(r'$f(x)$') 
        plt.xlim([-50, 5000]) 
#         plt.ylim([-10, 50]) 
        
    else: 
        plt.ylabel(r'$\| x \|$') 
        plt.xlim([-100, 10000]) 
#         plt.ylim([0, 15]) 

    plt.tight_layout() 
    
    if (small and f): 
        plt.savefig( '../figures/fig_f_eta{}_expsmall_sample_log.png'.format(eta) , dpi = 150)
    elif (small and (not f)): 
        plt.savefig( '../figures/fig_x_eta{}_expsmall_sample_log.png'.format(eta) , dpi = 150)
    elif ((not small) and (not f)): 
        plt.savefig( '../figures/fig_x_eta{}_explarge_sample_log.png'.format(eta) , dpi = 150)
    else: 
        plt.savefig( '../figures/fig_f_eta{}_explarge_sample_log.png'.format(eta) , dpi = 150)
        
    plt.clf() 


# In[14]:





# # in terms of sample complexity - linear scale

# In[15]:


plt.rcParams['font.size'] = 15 

def plot_sample(eta , f = True, small = True): 
    
    plt.rcParams['figure.figsize'] = 6,4

    for k in [1, 10, 20, 30]:  
        if small: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_expsmall'.format(k, eta),'rb')) 
        else: 
            res = pickle.load(open('../raw_data/res_k{0}_eta{1}_explarge'.format(k, eta),'rb')) 
        
        if f: 
            m = np.mean( np.array( res )[:,1,:] , axis = 0 )
            std = np.std( np.array( res )[:,1,:] , axis = 0 )
            m_aug = [  ] ; std_aug = []
            
            if k == 1:
                N = 11000
            else:
                N = 2000
    
            for i in range(N):
                m_aug = m_aug + [m[i]]*k*2
                std_aug = std_aug + [std[i]]*k*2
            
            plt.plot( m_aug , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m_aug)), np.array(m_aug) - np.array(std_aug), np.array(m_aug) + np.array(std_aug), alpha = 0.4  )
            
        else: 
            m = np.mean( np.array( res )[:,0,:] , axis = 0 )
            std = np.std( np.array( res )[:,0,:] , axis = 0 )
            
            m_aug = [  ] ; std_aug = []
            
            if k == 1:
                N = 11000
            else:
                N = 2000
            
            for i in range(N):
                m_aug = m_aug + [m[i]]*k*2
                std_aug = std_aug + [std[i]]*k*2
            
            plt.plot( m_aug , label = 'k = {}'.format(k) ) 
            plt.fill_between(range(len(m_aug)), np.array(m_aug) - np.array(std_aug), np.array(m_aug) + np.array(std_aug), alpha = 0.4 ) 

    plt.legend() 
    
#     plt.gca().set_yscale('log')

    plt.xlabel('Number of Function Evaluations') 
    
    if f: 
        plt.ylabel(r'$f(x)$') 
        plt.xlim([-50, 5000]) 
#         plt.ylim([-10, 50]) 
        
    else: 
        plt.ylabel(r'$\| x \|$') 
        plt.xlim([-100, 10000]) 
#         plt.ylim([0, 15]) 

    plt.tight_layout() 
    
    if (small and f): 
        plt.savefig( '../figures/fig_f_eta{}_expsmall_sample.png'.format(eta) , dpi = 150)
    elif (small and (not f)): 
        plt.savefig( '../figures/fig_x_eta{}_expsmall_sample.png'.format(eta) , dpi = 150)
    elif ((not small) and (not f)): 
        plt.savefig( '../figures/fig_x_eta{}_explarge_sample.png'.format(eta) , dpi = 150)
    else: 
        plt.savefig( '../figures/fig_f_eta{}_explarge_sample.png'.format(eta) , dpi = 150)
        
    plt.clf() 


# In[16]:

plot_sample_log(0.005 , f = True, small = True) 
plot_sample_log(0.005 , f = False, small = True) 
plot_sample_log(0.005 , f = True, small = False) 
plot_sample_log(0.005 , f = False, small = False) 


plot_sample(0.005 , f = True, small = True) 
plot_sample(0.005 , f = False, small = True) 
plot_sample(0.005 , f = True, small = False) 
plot_sample(0.005 , f = False, small = False) 


# In[ ]:





# In[ ]:





# In[ ]:




