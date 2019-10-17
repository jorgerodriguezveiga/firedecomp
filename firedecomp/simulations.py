"""Module with simulations function."""

# Python packages
import itertools
import logging as log
import argparse as _argparse
import pkg_resources
import sys
import pathlib
import os

# Package modules
from firedecomp.data import examples
from firedecomp.utilities import write
from firedecomp.utilities import utils


# simulations -----------------------------------------------------------------
def simulations(
        num_simulations=10,
        seeds=None,
        num_brigades=None,
        num_aircraft=None,
        num_machines=None,
        num_periods=None,
        modes=None,
        solver_options=None,
        solution_file='solution.csv',
        append_results=False
):
    """Solve simulations.

    Args:
        num_simulations (:obj:`int`): number of simulations of each case.
            Defaults to 10.
        seeds (:obj:`list`): list of seeds (integers). If None consider
            num_simulations argument.
        num_brigades (:obj:`list`): list with number of brigades. If ``None``
            defaults to ``[5, 10, 20]``.
        num_aircraft (:obj:`list`): list with number of aircraft. If ``None``
            defaults to ``[5, 10, 20]``.
        num_machines (:obj:`list`): list with number of machines. If ``None``
            defaults to ``[5, 10, 20]``.
        num_periods (:obj:`list`): list with number of periods. If ``None``
            defaults to``[20, 30, 40]``.
        modes (:obj:`str`): list of execution modes. If ``None`` defaults to
            ``['original', 'fix_work']``.
        solver_options (:obj:`dict`): solver options. If None default options.
        solution_file (:obj:`str`): filename (.csv) with transferring results.
            Defaults to ``'solution.csv'``.
        append_results (:obj:`str`): if ``True`` append results to the previous
            ones.
    """
    if num_simulations is None:
        num_simulations = 10

    if seeds is None:
        seeds = range(num_simulations)

    if num_brigades is None:
        num_brigades = [5, 10, 20]

    if num_aircraft is None:
        num_aircraft = [5, 10, 20]

    if num_machines is None:
        num_machines = [5, 10, 20]

    if num_periods is None:
        num_periods = [20, 30, 40]

    if modes is None:
        modes = ['original', 'fix_work']

    comb = itertools.product(
        num_brigades, num_aircraft, num_machines, num_periods)

    header = True

    if append_results:
        mode = 'a'
    else:
        mode = 'w'

    for brigades, aircraft, machines, periods in comb:
        log.info("#####################################")
        log.info("Number of brigades: {}".format(brigades))
        log.info("Number of aircraft: {}".format(aircraft))
        log.info("Number of machines: {}".format(machines))
        log.info("Number of periods: {}".format(periods))
        log.info("#####################################")
        for i in seeds:
            log.info("=====================================")
            log.info("Simulation: {}".format(i))
            log.info("=====================================")

            exec_info = {
                'seed': i,
                'num_brigades': brigades,
                'num_aircraft': aircraft,
                'num_machines': machines,
                'num_periods': periods
            }

            instance = examples.input_example(
                num_brigades=brigades, num_aircraft=aircraft,
                num_machines=machines, num_periods=periods,
                contention_factor=0.5,
                random=False, seed=i)

            for m in modes:
                log.info("-------------------------------------")
                log.info("Execution mode: {}".format(m))
                log.info("-------------------------------------")

                solution_dict = exec_info.copy()
                solution_dict['mode'] = m

                # Solver options
                orig_options = {}
                original_scip_options = {}
                fix_work_options = {}
                benders_scip_options = {}
                gcg_scip_options = {}

                if solver_options is not None:
                    if 'original' in solver_options:
                        orig_options = solver_options['original']
                        if 'solver_options' not in orig_options:
                            orig_options['solver_options'] = {}
                        if 'valid_constraints' not in orig_options:
                            orig_options['valid_constraints'] = None

                    if 'original_scip' in solver_options:
                        original_scip_options = solver_options['original_scip']

                    if 'fix_work' in solver_options:
                        fix_work_options = solver_options['fix_work']

                    if 'benders_scip' in solver_options:
                        benders_scip_options = solver_options['benders_scip']

                    if 'gcg_scip' in solver_options:
                        gcg_scip_options = solver_options['gcg_scip']

                    if 'AL' in solver_options:
                        AL_options = solver_options['AL']

                    if 'LR' in solver_options:
                        LR_options = solver_options['LR']

                instance.solve(
                    method=m,
                    original_options=orig_options,
                    original_scip_options=original_scip_options,
                    fix_work_options=fix_work_options,
                    benders_scip_options=benders_scip_options,
                    gcg_scip_options=gcg_scip_options,
                    AL_options=AL_options,
                    LR_options=LR_options,
                    min_res_penalty=1000000,
                    log_level=None)

                solution_dict.update(instance.get_solution_info())
                write.write_solution_as_csv(
                    solution_dict, header=header, mode=mode, file=solution_file,
                    dec=",")

                header = False
                mode = 'a'
# --------------------------------------------------------------------------- #


# parse_args ------------------------------------------------------------------
def parse_args():
    """Argument parser function (see argparse_).

    .. _argparse: http://newcoder.io/api/part-4/
    """
    # general parser and info
    parser = _argparse.ArgumentParser(
        prog='firedecomp_simulations',
        description="Tool to generate firedecomp simulations.",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n",
        "--num_simulations",
        type=int,
        default=10,
        help="Number of simulations of each case."
    )

    parser.add_argument(
        "-s",
        "--seeds",
        default=None,
        nargs='+',
        type=int,
        help="List with seeds. If None: 0 1 2 3 4 5 6 7 8 9."
    )

    parser.add_argument(
        "-nb",
        "--num_brig",
        default=None,
        nargs='+',
        type=int,
        help="List with the number of brigades. If None: 5 10 20."
    )

    parser.add_argument(
        "-na",
        "--num_air",
        default=None,
        nargs='+',
        type=int,
        help="List with the number of aircraft. If None: 5 10 20."
    )

    parser.add_argument(
        "-nm",
        "--num_mach",
        default=None,
        nargs='+',
        type=int,
        help="List with the number of machines. If None: 5 10 20."
    )

    parser.add_argument(
        "-np",
        "--num_periods",
        default=None,
        nargs='+',
        type=int,
        help="List with the number of periods. If None: 20 30 40."
    )

    parser.add_argument(
        "-m",
        "--modes",
        default=None,
        nargs='+',
        type=str,
        choices=['original', 'original_scip', 'fix_work', 'benders_scip',
                 'gcg_scip', 'AL', 'LR'],
        help="List of execution modes. "
             "Options allowed: original fix_work. "
             "If None: original fix_work."
    )

    parser.add_argument(
        "-so",
        "--solver_options",
        default=None,
        help="Solver options file."
    )

    parser.add_argument(
        "-pso",
        "--print_solver_options",
        action="store_true",
        help="Print solver options file.",
        required=False
    )

    parser.add_argument(
        "-o",
        "--out_directory",
        default='.',
        help="Output directory to store csv files."
    )

    parser.add_argument(
        "-sf",
        "--solution_file",
        default='solution.csv',
        help="Solution file (.csv)."
    )

    parser.add_argument(
        "-ar",
        "--append_results",
        action="store_true",
        help="Append results to the previous ones.",
        required=False
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count", default=0,
        help="Increases log verbosity for each occurence."
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="package version.",
        required=False
    )

    args = parser.parse_args()

    return args
# --------------------------------------------------------------------------- #


# logger ----------------------------------------------------------------------
def logger(debug):
    """Logging function.

    Args:
        debug (:obj:`bool`): if ``True`` logging level is debug.
    """
    # =========================================================================
    # Logging levels
    # -------------------------------------------------------------------------
    # log.debug('debug message')
    # log.info('info message')
    # log.warn('warn message')
    # log.error('error message')
    # log.critical('critical message')
    # -------------------------------------------------------------------------

    if debug is True:
        log_level = 10
    elif debug is False:
        log_level = 50
    elif isinstance(debug, int):
        if debug == 1:
            log_level = 30
        elif debug == 2:
            log_level = 20
        elif debug >= 3:
            log_level = 10
        else:
            log_level = 50
    elif isinstance(debug, str):
        if debug == "fix_work":
            log_level = 60
            logger = log.getLogger('fix_work')
            logger.setLevel(log_level)
            if len(logger.handlers) == 0:
                ch = log.StreamHandler()
                ch.setLevel(log_level)
                # create formatter and add it to the handlers
                formatter = log.Formatter("%(message)s")
                ch.setFormatter(formatter)
                logger.addHandler(ch)
            return
    else:
        log_level = 50

    # create log with 'my_logger'
    log.basicConfig(stream=sys.stdout, level=log_level,
                    format='%(message)s')

    # logg = log.getLogger()
    # logg.setLevel(log_level)
# --------------------------------------------------------------------------- #


# main ------------------------------------------------------------------------
def main():
    """Read data and execute the :func:`prepims.pripims`.

    .. _argparse: http://newcoder.io/api/part-4/
    """
    # =========================================================================
    # Load input
    # -------------------------------------------------------------------------
    args = parse_args()
    num_simulations = args.num_simulations
    seeds = args.seeds
    num_brig = args.num_brig
    num_air = args.num_air
    num_mach = args.num_mach
    num_periods = args.num_periods
    modes = args.modes
    solver_options_file = args.solver_options
    print_solver_options = args.print_solver_options
    out_directory = args.out_directory
    solution_file = args.solution_file
    append_results = args.append_results
    verbose = args.verbose
    version = args.version

    if version:
        try:
            package_info = pkg_resources.get_distribution(
                'firedecomp')
        except pkg_resources.DistributionNotFound:
            sys.exit(1)

        print(package_info.version)
        sys.exit(0)

    file_dir = utils.get_file_directory(__file__)
    solver_conf_path = 'config/solver_configuration.yaml'
    abs_solver_conf_path = utils.join_paths(file_dir, solver_conf_path)

    if print_solver_options:
        print(open(abs_solver_conf_path, 'r').read())
        sys.exit(0)

    if solver_options_file is None:
        solver_options = utils.load_yaml(abs_solver_conf_path)
    else:
        solver_options = utils.load_yaml(solver_options_file)

    # =========================================================================
    # Program
    # -------------------------------------------------------------------------

    # =========================================================================
    # Logging
    # -------------------------------------------------------------------------
    logger(verbose)

    # =========================================================================
    # Create out directory
    # -------------------------------------------------------------------------
    pathlib.Path(out_directory).mkdir(
        parents=True, exist_ok=True)
    solution_file_path = os.path.join(out_directory, solution_file)

    # =========================================================================
    # Execute the function
    # -------------------------------------------------------------------------
    simulations(
        num_simulations=num_simulations,
        seeds=seeds,
        num_brigades=num_brig,
        num_aircraft=num_air,
        num_machines=num_mach,
        num_periods=num_periods,
        modes=modes,
        solver_options=solver_options,
        solution_file=solution_file_path,
        append_results=append_results
    )
# --------------------------------------------------------------------------- #


if __name__ == '__main__':
    main()
