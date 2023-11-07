# rdeFBAvariability
This repository was built as part as the master's thesis 'Solving r-deFBA models of E.coli Core'.

The repository contains:
*   A copy of `PyrrrateFBA` which is used for most applications
*  `rdeFBA-Variability/factories` contains all scripts that were used to generate the results from the thesis
*  `rdeFBA-Variability/src` contains a several things
   * `src/models/` contains all models and their simulation scenarios
   * `src/model_scaling` defines a function that creates all objects needed to scale the *E.coli* Core model.
   * `src/optimization_problem` defines a class `rdeFBA_Problem` that uses PyrrrateFBA to solve models. It was mainly 
      built to seperate the construction from the MILP from the optimization of the MILP. Most notably, r-deFVA is implemented as part of this class. **Note: r-deFVA only works with CPLEX!**
* `solve_rdeFBA_problems.ipynb` is a user-friendly jupyter notebook that can be used to solve r-deFBA problems and and explore their varibility, using r-deFVA