# AW-MEP
Adaptive Weighted Multi-Expression Programmin

# AW-MEP  
Adaptive Weighted Multi-Expression Programming for Symbolic Regression  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/tsingke/AW-MEP)  
[![GitHub stars](https://img.shields.io/github/stars/tsingke/AW-MEP.svg)](https://github.com/tsingke/AW-MEP/stargazers)  



## 🚀 Project Overview  
Symbolic regression focuses on discovering **interpretable mathematical expressions** to model complex nonlinear relationships in data. Traditional Multi-Expression Programming (MEP) is elegant, yet often limited by premature convergence, fixed operator schemes, and sub-optimal exploration–exploitation balance.  

To overcome these limitations, we introduce **Adaptive Weighted Multi-Expression Programming (AW-MEP)** — a novel framework that integrates multiple self-regulating mechanisms for dynamic control of evolution. AW-MEP is designed to deliver improved convergence, higher generalization, and enhanced search efficiency for real-world symbolic modeling tasks.  

## 🔍 Key Innovations  
1. **Dynamic Operator Weighting Strategy**: Continuously assesses and updates the importance of genetic operators based on contribution feedback, guiding search direction and eliminating redundant operations.  
2. **Entropy-Guided Adaptive Mutation Rate**: Maintains population diversity and avoids stagnation by self-adjusting mutation intensity in response to entropy feedback.  
3. **Weighted Crossover + Elitist Preservation**: A refined genetic manipulation scheme combining weighted recombination and elitist retention to accelerate convergence without losing diversity.  
4. **Volcanic Simulated Annealing (VSA)**: Periodic energy-based perturbation enables the population to escape local optima and improves global search capability.  

## 📊 Experimental Highlights  
> Extensive experiments on 12 benchmarks (6 real-world regression datasets such as Boston Housing, California Housing, Diabetes, Red Wine Quality, Airfoil Self-Noise, Concrete Compressive Strength; along with 6 symbolic regression functions) demonstrate that AW-MEP consistently outperforms classical MEP, Gene Expression Programming (GEP), and standard symbolic regression (SR) methods.  
> Quantitative gains: Mean Squared Error (MSE) reduction of **7%–47%**, up to **30% faster convergence**, and lower computational cost.  

## 🧩 Getting Started  
### ✅ Prerequisites  
- MATLAB R2020a (or later)  
- Symbolic Math Toolbox (optional for expression simplification)  
- Basic familiarity with symbolic regression and evolutionary programming  

### 🛠 Installation  
```bash
git clone https://github.com/tsingke/AW-MEP.git  
cd AW-MEP
```

## 🤝 Contributing

Contributions (bug reports, feature requests, pull-requests) are welcomed! Please read the CONTRIBUTING.md
 guidelines before submitting.

##  📄 License

This project is licensed under the MIT License — see the LICENSE
 file for details.
