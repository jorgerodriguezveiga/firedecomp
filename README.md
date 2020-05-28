# firedecomp

firedecomp package provide different decomposition methods to solve the wildfire containment problem.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The following prerequisites are needed:
+ **gurobipy**: the main requirement is to have installed the [gurobi](http://www.gurobi.com/registration/general-reg) software and its api ([gurobipy](http://www.gurobi.com/documentation/)).

The remain requirements will be installed with the installation of **firedecomp** package.

#### Software

The following software is needed:

+ **GUROBI**: Register and install the solver [gurobi](http://www.gurobi.com/registration/general-reg).
+ **SCIP**: download and install the software [scip](https://scip.zib.de/index.php#download).
+ **GCG**: only available on Linux [GCG](http://gcg.or.rwth-aachen.de/doc/INSTALL.html).

#### Python packages.

The python package requirements are stored in [requirements.txt](requirements.txt) file. To install them type:
```
pip install -r requirements.txt
```

Also gurobipy package is needed, to install it follow the [instructions](http://www.gurobi.com/documentation/7.5/quickstart_mac/the_gurobi_python_interfac.html).

## Installing

To install the firedecomp package, type:
```
pip install git+http://github.com/jorgerodriguezveiga/firedecomp.git
```

## Execution

### Simulations

To execute firedecomp simulations the ``firedecomp_simulations`` command line has been created.

To see the function help, type:
```
firedecomp_simulations -h
```

If ``firedecomp_simulations`` function is executed with no input arguments default simulations will be executed.

