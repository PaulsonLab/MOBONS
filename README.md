# MOBONS: A Network-Based Approach to Multi-Objective Optimization

![Network System](data_plot_MOBO/network_system.png)

*Illustration of a complex network system integrating process simulation, CFD, life cycle analysis (LCA), ecological modeling, and economic evaluation.*

## Overview
Designing modern industrial systems requires balancing competing objectives such as **profitability, resilience, and sustainability**, while accounting for complex **technological, economic, and environmental** interactions. Multi-objective optimization (MOO) methods help navigate these trade-offs, but selecting an appropriate solver is challenging, especially when system representations vary from **white-box (equation-based)** to **black-box (data-driven)** models. 

**MOBONS** is a novel **Bayesian optimization-inspired algorithm** that unifies **grey-box MOO** through **network representations**, enabling flexible modeling of interconnected systems. Unlike traditional approaches, MOBONS:
- Supports **cyclic dependencies**, allowing feedback loops, recycle streams, and multi-scale simulations.
- **Incorporates constraints** while maintaining the sample efficiency of Bayesian optimization.
- Enables **parallel evaluations** to improve convergence speed.
- Leverages **network structure** for **scalability** beyond conventional MOO solvers.

## Installation of dependencies:

To install the required packages, run the following command:
pip install -r requirements.txt


## Case Studies
At this time, our scripts only support the limiting case of handling **directed acyclic graph (DAG)**, future release will contain extensions to generalized formulations from the paper.
MOBONS is demonstrated on two case studies:

### 1. Synthetic ZDT4 Benchmark
![ZDT4 Case Study](data_plot_MOBO/ZDT4_case_study.png)

A widely used synthetic benchmark [1] for testing MOO algorithms. MOBONS effectively optimizes **discontinuous, multi-modal landscapes** where traditional solvers struggle. A comparison of the baseline algorithms [2,3] with MOBONS is performed to demonstrate the effectiveness of the network system perspective.

- **Simulation Code:** [`test_func_MOBO/ZDT4_Simulation.py`](test_func_MOBO/ZDT4_Simulation.py)

### 2. Sustainable Ethanol Production
![Ethanol Case Study](data_plot_MOBO/Ethanol_case_study.png)

This case models **bioethanol production** by integrating **process simulation, and economic evaluation** to optimize sustainability metrics. MOBONS outperforms conventional solvers by efficiently handling **interdependent process models and dynamic trade-offs**.

- **Simulation Code:** [`test_func_MOBO/Ethanol_Fermentation_Simulation.py`](test_func_MOBO/Ethanol_Fermentation_Simulation.py)

## Running the Case Studies
To run the case studies, set an appropriate value of the variable `example_name` in the script [`performance_test_MOBO.py`](performance_test_MOBO.py). 

The **network gaussian process** for all the case studies in this work is defined in [`Objective_FN_MOBO.py`](Objective_FN_MOBO.py).

## Citation
If you use MOBONS in your work, please cite our chapter. More details can be found in our full publication. Details coming soon.

## References

[1] Eckart Zitzler, Kalyanmoy Deb, and Lothar Thiele. Comparison of multiobjective evolutionary
algorithms: Empirical results. Evolutionary Computation, 8(2):173–195, June 2000.

[2] Samuel Daulton, Maximilian Balandat, and Eytan Bakshy. Differentiable expected hypervolume
improvement for parallel multi-objective bayesian optimization. 2020.

[3] S. Ashwin Renganathan and Kade E. Carlson. qpots: Efficient batch multiobjective bayesian opti-
mization via pareto optimal thompson sampling, 2023.

---
For further details, please refer to the documentation or contact us!

