"""Module with classes examples."""

# Python packages
import numpy as np
import bson

# Package modules
from firedecomp.classes.resources import Resources, Resource
from firedecomp.classes.wildfire import Wildfire, Period
from firedecomp.classes.groups import Group, Groups
from firedecomp.classes.resources_wildfire import ResourcePeriod, ResourcesWildfire
from firedecomp.classes.groups_wildfire import GroupPeriod, GroupsWildfire
from firedecomp.classes.problem import Problem
from . import random as r


# INPUT =======================================================================

# input_example ---------------------------------------------------------------
def input_example(num_brigades=5, num_aircraft=5, num_machines=5,
                  num_periods=20, ini_perimeter=20, random=False, seed=None):
    """Input example."""
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = 1
        if random is False:
            np.random.seed(seed)

    brigades = resources_example(num_brigades=num_brigades,
                                 num_aircraft=0,
                                 num_machines=0,
                                 random=random, seed=seed)

    aircraft = resources_example(num_brigades=0,
                                 num_aircraft=num_aircraft,
                                 num_machines=0,
                                 random=random, seed=seed)

    machines = resources_example(num_brigades=0,
                                 num_aircraft=0,
                                 num_machines=num_machines,
                                 random=random, seed=seed)

    resources = Resources(list(brigades) + list(aircraft) + list(machines))

    brigades_grp = Group(name='brigades', resources=brigades)
    aircraft_grp = Group(name='aircraft', resources=aircraft)
    machines_grp = Group(name='machines', resources=machines)
    groups = Groups([brigades_grp, aircraft_grp, machines_grp])

    wildfire = wildfire_example(
        num_periods=num_periods, ini_perimeter=ini_perimeter, random=random,
        seed=seed)

    res_wild = ResourcesWildfire([ResourcePeriod(i, p, resources_efficiency=1)
                                  for i in resources for p in wildfire])

    grp_wild = GroupsWildfire([
        GroupPeriod(g, p,
                    min_res_groups=min(2, g.size()), max_res_groups=g.size())
        if g.name == "brigades" else
        GroupPeriod(g, p, min_res_groups=0, max_res_groups=g.size())
        for g in groups for p in wildfire])

    return Problem(resources, wildfire, groups, grp_wild, res_wild)
# --------------------------------------------------------------------------- #


# RESOURCES ===================================================================

# resource_example -----------------------------------------------------------
def resource_example(random=False, res_type='brigade', seed=None):
    """Resource class example.

    Args:
        random (:obj:`bool`): if True the resource is generated randomly.
            Defaults to ``False``.
        res_type (:obj:`str`): resource type. Options allowed: ``'aircraft'``,
            ``'brigade'``, ``'machine'``. Defaults to ``'aircraft'``.
        seed (:obj:`int`): numpy seed. If None no seed. Defaults to ``None``.
    """
    if seed is not None:
        np.random.seed(seed)
        if random is False:
            res_id = str(seed)
        else:
            res_id = str(bson.ObjectId())

    else:
        seed = 1
        if random is False:
            np.random.seed(seed)
            res_id = str(seed)
        else:
            res_id = str(bson.ObjectId())

    if res_type == 'brigade':
        name = 'brigade_{}'.format(res_id)
        working_this_wildfire = np.random.choice([True, False], p=[0.1, 0.9])
        working_other_wildfire = not working_this_wildfire \
                                 and np.random.choice([True, False],
                                                      p=[0.3, 0.7])
        if working_this_wildfire is True:
            arrival = 0
            work = r.random_num(0, 470)
            rest = 0
        else:
            arrival = r.random_num(30, 120)
            if working_other_wildfire is True:
                work = r.random_num(60, 400)
                rest = 0
            else:
                work = 0
                rest = 0

        total_work = work
        performance = r.random_num(2, 10)
        fix_cost = r.random_num(0, 1000)
        variable_cost = r.random_num(200, 500)
        time_between_rests = 10
        max_work_time = 480
        necessary_rest_time = 0
        max_work_daily = 480

    elif res_type == 'machine':
        name = 'machine_{}'.format(res_id)
        working_this_wildfire = np.random.choice([True, False], p=[0.1, 0.9])
        working_other_wildfire = not working_this_wildfire \
                                 and np.random.choice([True, False],
                                                      p=[0.3, 0.7])
        if working_this_wildfire is True:
            arrival = 0
            work = r.random_num(0, 470)
            rest = 0
        else:
            arrival = r.random_num(60, 180)
            if working_other_wildfire is True:
                work = r.random_num(60, 400)
                rest = 0
            else:
                work = 0
                rest = 0

        total_work = work
        performance = r.random_num(4, 15)
        fix_cost = r.random_num(0, 1000)
        variable_cost = r.random_num(500, 1000)
        time_between_rests = 10
        max_work_time = 480
        necessary_rest_time = 0
        max_work_daily = 480

    else:
        name = 'aircraft_{}'.format(res_id)
        working_this_wildfire = np.random.choice([True, False], p=[0.1, 0.9])
        working_other_wildfire = not working_this_wildfire \
                                 and np.random.choice([True, False],
                                                      p=[0.3, 0.7])
        if working_this_wildfire is True:
            arrival = 0
            work = r.random_num(0, 120)
            rest = np.random.choice([0, 10, 20, 30], p=[0.7, 0.1, 0.1, 0.1])
        else:
            arrival = r.random_num(10, 40)
            if working_other_wildfire is True:
                rest = np.random.choice([0, 10, 20, 30],
                                        p=[0.7, 0.1, 0.1, 0.1])
                if rest > 0:
                    work = 120 + rest
                else:
                    work = r.random_num(0, 120)
            else:
                work = 0
                rest = 0

        total_work = 160*np.random.choice([0, 1, 2]) + work
        performance = r.random_num(4, 15)
        fix_cost = r.random_num(0, 1000)
        variable_cost = r.random_num(1000, 3000)
        time_between_rests = 10
        max_work_time = 120
        necessary_rest_time = 40
        max_work_daily = 480

    return Resource(name=name,
                    working_this_wildfire=working_this_wildfire,
                    working_other_wildfire=working_other_wildfire,
                    arrival=arrival,
                    work=work,
                    rest=rest,
                    total_work=total_work,
                    performance=performance,
                    fix_cost=fix_cost,
                    variable_cost=variable_cost,
                    time_between_rests=time_between_rests,
                    max_work_time=max_work_time,
                    necessary_rest_time=necessary_rest_time,
                    max_work_daily=max_work_daily)
# --------------------------------------------------------------------------- #


# resources_example ------------------------------------------------------------
def resources_example(num_brigades=5, num_aircraft=5, num_machines=5,
                      random=False, seed=None):
    """Resources class example.

    Args:
        num_brigades (:obj:`int`): number of brigades. Defaults to ``5``.
        num_aircraft (:obj:`int`): number of aircraft. Defaults to ``5``.
        num_machines (:obj:`int`): number of machines. Defaults to ``5``.
        random (:obj:`bool`): if True the resource is generated randomly.
            Defaults to ``False``.
        seed (:obj:`int`): numpy seed. If None no seed. Defaults to ``None``.
    """
    if seed is None:
        seed = 0
    resources = ['brigade']*num_brigades + \
                ['aircraft']*num_aircraft + \
                ['machine']*num_machines

    return Resources([
            resource_example(random=random, res_type=rec, seed=seed+i)
            for i, rec in enumerate(resources)])
# --------------------------------------------------------------------------- #


# WILDFIRE ====================================================================

# wildfire_example ------------------------------------------------------------
def wildfire_example(num_periods=10, ini_perimeter=10, random=False,
                     seed=None):
    """Wildfire class example.

    Args:
        num_periods (:obj:`int`): number of periods.
        ini_perimeter (:obj:`float`): start perimeter.
        random (:obj:`bool`): if True the period is generated randomly.
            Defaults to ``False``.
        seed (:obj:`int`): numpy seed. If None no seed. Defaults to ``None``.
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = 1
        if random is False:
            np.random.seed(seed)

    perimeter = ini_perimeter
    cost = r.random_num(300*perimeter, 500*perimeter, zero=-2)

    periods_list = [Period(name=0, perimeter=perimeter, cost=cost)]

    for p in range(1, num_periods):
        inc_per = r.random_num(max(1, perimeter/30),
                               min(3, perimeter/10),
                               zero=-2)
        perimeter += inc_per
        cost += r.random_num(300*inc_per, 500*inc_per)
        periods_list.append(Period(name=p, perimeter=perimeter, cost=cost))
    return Wildfire(periods_list, time_per_period=10)
# --------------------------------------------------------------------------- #


# period_example --------------------------------------------------------------
def period_example():
    """Period class example."""
    return Period(1, increment_perimeter=150, increment_cost=100)
# --------------------------------------------------------------------------- #
