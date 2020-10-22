Change Log
==========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org/).


[Unreleased]
------------

v4.1.0 - 22/10/2020
-------------------

### Added ###
- Add new layouts for the paper figures.

### Changed ###
- Improve analysis and plots of the results of the simulations.


v4.0.0 - 28/05/2020
-------------------

### Added ###
- Final results of simulations.

### Changed ###
- Improve analysis and plots of the results of the simulations.


v3.2.0 - 08/10/2019
-------------------
### Added ###
- Add --seed (-s) argument to specify the desired seeds to solve.


v3.1.2 - 01/10/2019
-------------------
### Added ###
- Add new fix_work results.

### Changed ###
- Improve performance profile plot function in simulations.

### Fixed ###
- Fix error in gcg_scip.

v3.1.1 - 24/09/2019
-------------------
### Fixed ###
- Fix error getting start resource information.
- Fix error in fix_work algorithm.

v3.1.0 - 19/08/2019
-------------------

### Added ###
- Display fix_work algorithm behaviour.
- Add method to get initial solution. If an initial solution is not founded the function returns False, otherwise True.

### Changed ###
- Updated default solver options.

### Fixed ###
- Fixed problems in fix_work algorithm.
- Fix error in scheduling plot.
- Fix error updating fix_work solver options.

v3.0.0 - 29/05/2019
-------------------

### Added ###
- Add MANIFEST.in.
- Add original_scip solve method.
- Add benders_scip solve method.
- Add gcg_scip solve method.

### Changed ###
- Update documentation.
- Update instances generation to try to avoid infeasible problems.

### Fixed ###
- Fix error when no solution is found.
- Fix error writing None object in csv.
- Fix error try to write results when there are not.

v2.0.0 - 01/05/2019
-------------------

### Added ###
- Add notebook to represent simulations information.
- Add timelimit to benders method option.
- Use initial solution and add max_obj constraint in the iteration period method to improve solving time.
- Add command line argument to append results instead of generate a new results file.

### Changed ###
- Change benders method name to fix_work.
- Improve instances generation.

### Removed ###

### Fixed ###
- Fix error getting variable value, now if solver returns 0.9999 it si considered like 1.

1.1.0 - 2019-02-22
------------------

### Added ###
- Add options to solve the original feasible model adding slack variables to wildfire containment constraints.
- Add simulations folder.

### Changed ###
- Round to 3 digits model information to avoid numerical problems.

### Removed ###

### Fixed ###
- Fix error computing rests periods.


1.0.0 - 2019-02-18
------------------

### Added ###
+ Add reformulation
+ Add command line option to execute simulations.

### Changed ###
+ Update classes structure.

### Removed ###

### Fixed ###
+ Fix model errors.


0.1.0 - 2019-01-14
------------------

### Added ###
- Package creation.

### Changed ###

### Removed ###

### Fixed ###