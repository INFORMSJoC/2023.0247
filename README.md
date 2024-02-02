[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Convergence Rates of Zeroth-order Gradient Descent (ZGD) for Łojasiewicz Functions

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE). This is a code repo for experiments in the paper "Convergence Rates of Zeroth-order Gradient Descent for Łojasiewicz Functions'' by Tianyu Wang and Yasong Feng.


The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Convergence Rates of Zeroth-order Gradient Descent for Łojasiewicz Functions](https://doi.org/) by Tianyu Wang and Yasong Feng. 

**Important: This code is a frozen copy of 
https://github.com/wangt1anyu/code-zeroth-order-Lojasiewicz. Please go there if you would like to
get an author-maintained version or would like support**

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0247

https://doi.org/10.1287/ijoc.2023.0247.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@misc{wf2024loj,
  author =        {Tianyu Wang and Yasong Feng},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Convergence Rates of Zeroth-order Gradient Descent for Łojasiewicz Functions}},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0247.cd},
  url =           {https://github.com/INFORMSJoC/2023.0247},
  note =          {Available for download at https://github.com/INFORMSJoC/2019.0000}
}  
```

## Replicating

- To run the code, you will need to make sure that you have already installed [Anaconda3](https://www.anaconda.com/). 

The python script "exp.py" and python notebook "exp.ipynb" contain codes for experiments in the main text of the paper, and the files "exp-linesearch.nb" and "exp-linesearch.ipynb" contain codes for experiments in the appendix of the paper. 

The following code block is an example for running the Zeroth-order Gradient Descent (ZGD) algorithm on a Łojasiewicz function. 

```
python exp.py --plain_gd 0 --large 0 --k 10 
```

- "plain_gd" should be either 0 ot 1. When plain_gd is 1, the program runs plain Gradient Descent(GD) on a Łojasiewicz function. When plain_gd is 0, the program runs ZGD on a Łojasiewicz function. The Łojasiewicz function here is either $F_1$ or $F_2$ defined in the paper, depending on the value of argument "large".
- "large" should be either 0 or 1. When large is 1, the objective function is $F_1$ defined in the paper. When large is 0, the objective function is $F_2$ defined in the paper.
- "k" should be one of {1,10,20,30}. k is the number of random directions used in the algorithm.

The following code block is an example for running the Zeroth-order Gradient Descent (ZGD) with line search on a Łojasiewicz function. This algorithm is detailed in the Appendix of the paper. 

```
python exp-linesearch.py --k 1 --large 0 
```

- "large" should be either 0 or 1. When large is 1, the objective function is $F_1$ defined in the paper. When large is 0, the objective function is $F_2$ defined in the paper.
- "k" should be one of {1,10,20,30}. k is the number of random directions used in the algorithm.

All the commands for experiments in the paper are contained in the "run.bash" file.
So to reproduce the results in the paper, please run 
```
bash run.bash
python plotting.py
python plotting-linesearch.py
```

The authors strongly recommend replicating the results in the following way. Directly executing "exp.ipynb" cell-by-cell and "exp-linesearch.ipynb" cell-by-cell will create raw data in the "raw_data" folder. After the raw data are in place, running "plotting.ipynb" cell-by-cell and "plotting-linesearch.ipynb" cell-by-cell will save plots in the "figures" folder. 


## Results

All results have been reported in the paper Section 6 and the Appendix. As an example, running the codes with large = True, eta = 0.005, and k = 1,10,20,30 gives the following figure. In the figure, lines labelled "k = 1" (resp., k = 10,20,30) plots the results of ZGD with k = 1 (resp. k=10,20,30), and the line labelled "GD" plots the results of gradient descent. 

![example](./example.png)

Users are highly recommended to use iterative python notebook for figure plotting. 

## Ongoing Development

This code is being developed on an on-going basis at the author-maintained 
[Github repo](https://github.com/wangt1anyu/code-zeroth-order-Lojasiewicz).

## Support

Please contact [Tianyu Wang](wangtianyu@fudan.edu.cn) or [Yasong Feng](ysfeng20@fudan.edu.cn) if you have any questions.
