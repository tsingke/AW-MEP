
# AW-MEP  
Adaptive Weighted Multi-Expression Programming for Symbolic Regression  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/tsingke/AW-MEP) [![GitHub stars](https://img.shields.io/github/stars/tsingke/AW-MEP.svg)](https://github.com/tsingke/AW-MEP/stargazers)  

**Title**: Adaptive Weighted Multi-Expression Programming

```
 Authors：Xin Yin, Qingke Zhang*, Junqing Li, Huaxiang Zhang
```
> 1. School of Information Science and Engineering, Shandong Normal University, Jinan 250358, China
> 
> 2. School of Information Science and Engineering, Yunnan Normal University,  Yunan 650500, China
> 
> 3. Aerospace InformationTechnology University, Jinan, 250200, China
>
> 
> Corresponding Author: **Qingke Zhang** ， Email: tsingke@sdnu.edu.cn ， Tel :  +86-13953128163

This paper is being considered for submission to the Elsevier journal

## Project Overview  
Symbolic regression focuses on discovering **interpretable mathematical expressions** to model complex nonlinear relationships in data. Traditional Multi-Expression Programming (MEP) is elegant, yet often limited by premature convergence, fixed operator schemes, and sub-optimal exploration–exploitation balance. To overcome these limitations, we introduce **Adaptive Weighted Multi-Expression Programming (AW-MEP)** — a novel framework that integrates multiple self-regulating mechanisms for dynamic control of evolution. AW-MEP is designed to deliver improved convergence, higher generalization, and enhanced search efficiency for real-world symbolic modeling tasks.  

## Key Innovations  
1. **Dynamic Operator Weighting Strategy**: Continuously assesses and updates the importance of genetic operators based on contribution feedback, guiding search direction and eliminating redundant operations.  
2. **Entropy-Guided Adaptive Mutation Rate**: Maintains population diversity and avoids stagnation by self-adjusting mutation intensity in response to entropy feedback.  
3. **Weighted Crossover + Elitist Preservation**: A refined genetic manipulation scheme combining weighted recombination and elitist retention to accelerate convergence without losing diversity.  
4. **Volcanic Simulated Annealing (VSA)**: Periodic energy-based perturbation enables the population to escape local optima and improves global search capability.  

## Experimental Highlights  
> Extensive experiments on 12 benchmarks (6 real-world regression datasets such as Boston Housing, California Housing, Diabetes, Red Wine Quality, Airfoil Self-Noise, Concrete Compressive Strength; along with 6 symbolic regression functions) demonstrate that AW-MEP consistently outperforms classical MEP, Gene Expression Programming (GEP), and standard symbolic regression (SR) methods.  
> Quantitative gains: Mean Squared Error (MSE) reduction of **7%–47%**, up to **30% faster convergence**, and lower computational cost.  


##  License

This project is licensed under the MIT License — see the LICENSE file for details.

 ## 🙏 Acknowledgements

**We would like to express our sincere gratitude to editors and the anonymous reviewers for taking the time to review our paper.** 

This work is supported by the National Natural Science Foundation of China (Grant Nos. 62006144) and the Taishan Scholar Project of Shandong Province, China (Grant No.ts20190924).
