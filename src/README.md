
# Documentation

All the commands for experiments in the paper are contained in the "run.bash" file. So to reproduce the results in the paper, please run

```
bash run.bash
python plotting.py
python plotting-linesearch.py
```

Please also see the comments in source code files. 

## In the file "exp.py"

- the function func_large is F_1 defined in the paper.  
- the function func_small is F_2 defined in the paper.  
- the function sample_stiefel returns a random k-frame.  
  - parameters: 
    - n: ambient dimension. k = 30 for all experiments in the paper. 
    - k: number of random directions. should be one of {1,10,20,30}. 
      
- the function get_grad_est returns a gradient estimator.  
  - parameters: 
    - func: the objective functions.
    - n: ambient dimension. k = 30 for all experiments in the paper.
    - k: number of random directions. should be one of {1,10,20,30}.
    - delta: finite difference granularity. 


- the function get_res returns results for ZGD on a Lojasiewicz function in log scale. 
  - parameters:
    - k: number of random directions. should be one of {1,10,20,30}.
    - eta: learning rate, learning rate is set to 0.005 for all experiments in the paper.
    - ITER: number of total iterations.
    - large: when set to True, the objective function is $F_1$ defined in the paper. when set to False, the objective function is $F_2$ defined in the paper.
    - rep: repeat the experiments for "rep" number of times. By default, rep is set to 10. 



- the function get_res_gred returns results for GD on a Lojasiewicz function in log scale.
  - parameters:
    - eta: learning rate, learning rate is set to 0.005 for all experiments in the paper.
    - ITER: number of total iterations.
    - large: when set to True, the objective function is F_1 defined in the paper. when set to False, the objective function is $F_2$ defined in the paper.
    - rep: repeat the experiments for "rep" number of times. By default, rep is set to 10. 

## In the file "exp-linesearch.py"

- the function linesearch returns the next iterate point using binary linesearch.
  - parameters:
    - x: the current iterate.
    - eta: learning rate, learning rate is set to 0.005 for all experiments in the paper.
    - grad: the direction along which linesearch is performed.
    - eps: linesearch tolerance. If the distance between two consecutive points between the binary linesearch procedure is less then eps, then terminate.
    - large: when set to True, the objective function is $F_1$ defined in the paper. when set to False, the objective function is $F_2$ defined in the paper. 

- the function get_res_ls returns results for ZGD on a Lojasiewicz function in log scale.
  - parameters:
  - k: number of random directions. should be one of {1,10,20,30}.
  - eta: learning rate, learning rate is set to 0.005 for all experiments in the paper.
  - ITER: number of total iterations.
  - large: when set to True, the objective function is $F_1$ defined in the paper. when set to False, the objective function is $F_2$ defined in the paper.
  - rep: repeat the experiments for "rep" number of times. By default, rep is set to 10. 


## In the file "plotting.py"


- the function plot_iter_log plots the convergence results in terms of iterations in log scale
  - parameters:
    - eta: learning rate, should specify the learning rate used in the experiments. all learning rate is set to 0.005 for experiments in the paper. 
    - f: When set to True, the y-axis is set to function value $f(x)$. When set to False, the y-axis is set to $\| x - x^* \|$, which is $\| x \|$ for the experiments in the paper.
    - small: when set to True, the objective function is $F_2$ defined in the paper. when set to False, the objective function is $F_1$ defined in the paper. 


- the function plot_iter plots the convergence results in terms of iterations in linear scale
  - parameters:
    - eta: learning rate, should specify the learning rate used in the experiments. all learning rate is set to 0.005 for experiments in the paper. 
    - f: When set to True, the y-axis is set to function value $f(x)$. When set to False, the y-axis is set to $\| x - x^* \|$, which is $\| x \|$ for the experiments in the paper.
    - small: when set to True, the objective function is $F_2$ defined in the paper. when set to False, the objective function is $F_1$ defined in the paper. 


- the function plot_sample_log plots the convergence results in terms of sample in log scale
  - parameters:
    - eta: learning rate, should specify the learning rate used in the experiments. all learning rate is set to 0.005 for experiments in the paper. 
    - f: When set to True, the y-axis is set to function value $f(x)$. When set to False, the y-axis is set to $\| x - x^* \|$, which is $\| x \|$ for the experiments in the paper.
    - small: when set to True, the objective function is $F_2$ defined in the paper. when set to False, the objective function is $F_1$ defined in the paper.
   
- the function plot_sample plots the convergence results in terms of sample in linear scale 
  - parameters:
    - eta: learning rate, should specify the learning rate used in the experiments. all learning rate is set to 0.005 for experiments in the paper. 
    - f: When set to True, the y-axis is set to function value $f(x)$. When set to False, the y-axis is set to $\| x - x^* \|$, which is $\| x \|$ for the experiments in the paper.
    - small: when set to True, the objective function is $F_2$ defined in the paper. when set to False, the objective function is $F_1$ defined in the paper. 

## In the file "plotting-linesearch.py"

- the function plot_iter_log plots the convergence results (of ZGD with line search) in terms of iterations in log scale.
  - parameters:
    - eta: learning rate, learning rate is set to 0.005 for all experiments in the paper.
    - k: number of random directions. should be one of {1,10,20,30}.
    - f: When set to True, the y-axis is set to function value f(x). When set to False, the y-axis is set to $\| x - x^* \|$, which is $\| x \|$ for the experiments in the paper.
    - small: when set to True, the objective function is $F_2$ defined in the paper. when set to False, the objective function is $F_1$ defined in the paper. 

- the function plot_iter plots the convergence results (of ZGD with line search) in terms of iterations in linear scale.
  - parameters:
    - eta: learning rate, all learning rate is set to 0.005 for experiments in the paper.
    - k: number of random directions. should be one of {1,10,20,30}.
    - f: When set to True, the y-axis is set to function value f(x). When set to False, the y-axis is set to $\| x - x^* \|$, which is $\| x \|$ for the experiments in the paper.
    - small: when set to True, the objective function is $F_2$ defined in the paper. when set to False, the objective function is $F_1$ defined in the paper. 
