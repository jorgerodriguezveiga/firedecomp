import os
import pandas as pd
import plotly.graph_objs as go
import plotly
import operator


def get_qgrid_options():
    return {
        # SlickGrid options
        'fullWidthRows': True,
        'syncColumnCellResize': True,
        'forceFitColumns': False,
        'defaultColumnWidth': 100,
        'rowHeight': 28,
        'enableColumnReorder': True,
        'enableTextSelectionOnCells': True,
        'editable': True,
        'autoEdit': False,
        'explicitInitialization': True,

        # Qgrid options
        'maxVisibleRows': 10,
        'minVisibleRows': 10,
        'sortable': True,
        'filterable': True,
        'highlightSelectedCell': False,
        'highlightSelectedRow': True
    }


def read_csvs(folder='Finales', file='solution.csv'):
    subfolders = os.listdir(folder)
    list_df = []

    for subdir in subfolders:
        file_path = os.path.join(folder, subdir, file)
        try:
            list_df.append(
                pd.read_csv(file_path, sep=';', decimal=",", na_values="None"))
        except FileNotFoundError as e:
            print(e)

    df = pd.concat(list_df, ignore_index=True)
    df['cost'] = df["res_cost"] + df['wildfire_cost']
    df['num_resources'] = df['num_brigades'] + df['num_aircraft'] + df[
        'num_machines']

    return df


def get_best(df, columns=None, groupby=None, filter_best=None, best_prefix='best_',
             rel_prefix='rel_'):
    if groupby is None:
        groupby = ['num_brigades', 'num_aircraft', 'num_machines',
                   'num_periods', 'seed']

    if columns is None:
        columns = [{'name': 'obj_fun', 'sense': 'min'},
                   {'name': 'elapsed_time', 'sense': 'min'},
                   {'name': 'solve_time', 'sense': 'min'}]

    if filter_best is None:
        filter_best = []

    filter_df = df.copy()
    for condition in filter_best:
        op = getattr(operator, condition['operator'])
        filter_df = filter_df[op(filter_df[condition['column']], condition['value'])]

    sense_instances = pd.DataFrame()
    for column in columns:
        sense_instances[best_prefix + column['name']] = getattr(
            filter_df.groupby(groupby)[column['name']], column['sense'])()

    sense_instances = sense_instances.reset_index()
    sense_instances = sense_instances.rename(
        columns={column['name']: best_prefix + column['name'] for column in
                 columns})

    new_df = pd.merge(filter_df, sense_instances, on=groupby)

    for column in columns:
        best = new_df[best_prefix + column['name']]
        new_df[rel_prefix + column['name']] = new_df[column['name']] / best

    return new_df


def performance_profile_graph(
        df, scatter_by='mode', x='elapsed_time', conditions=None,
        groupby=None, columns=None, filter_best=None, best_prefix='best_',
        rel_prefix='rel_', npoints=500):
    """
    Args:
        df: DataFrame.
        scatter_by: mode
        x: column of the DataFrame.
        conditions: list of conditions.
            [{'column': 'rel_obj_fun', 'operator': 'eq', 'value': 1}]
        groupby: which columns define group index.
        columns: columns to decide how to compute best and rel.
        best_prefix: best prefix.
        rel_prefix: relative prefix.
        npoints: number of points in x axis.
    """
    if groupby is None:
        groupby = ['num_brigades', 'num_aircraft', 'num_machines',
                   'num_periods', 'seed']

    if columns is None:
        columns = [{'name': 'obj_fun', 'sense': 'min'},
                   {'name': 'solve_time', 'sense': 'min'},
                   {'name': 'elapsed_time', 'sense': 'min'}]

    if conditions is None:
        conditions = []

    df = get_best(
        df, groupby=groupby, columns=columns, filter_best=filter_best,
        best_prefix=best_prefix, rel_prefix=rel_prefix)

    num_instances = max(
        [sum(df['mode'] == m) for m in set(df['mode'])]
    )
    performance = dict()

    min_x = df[x].min() * 0.9
    max_x = df[x].max() * 1.1
    step = (max_x - min_x)/npoints
    x_range = [min_x + i * step for i in
               range(npoints)]

    for m in set(df[scatter_by]):
        performance[m] = dict()
        print("Scatter: {}".format(m))
        for t in x_range:
            m_df = df[df[scatter_by] == m]
            m_df = m_df[m_df[x] <= t]
            for condition in conditions:
                op = getattr(operator, condition['operator'])
                m_df = m_df[op(m_df[condition['column']], condition['value'])]
            performance[m][t] = m_df.shape[0]/num_instances

    performance_df = pd.DataFrame(performance)

    data = []

    for c in performance_df.columns:
        data.append(
            go.Scatter(x=performance_df.index, y=performance_df[c], name=c))

    plotly.offline.iplot(data)


def instance_comparison(
        df, scatter_by='mode', reference='original', x='elapsed_time',
        groupby=None, columns=None, best_prefix='best_',
        rel_prefix='rel_'):
    """
    Args:
        df: DataFrame.
        scatter_by: mode
        x: column of the DataFrame.
        groupby: which columns define group index.
        columns: columns to decide how to compute best and rel.
        best_prefix: best prefix.
        rel_prefix: relative prefix.
    """
    if groupby is None:
        groupby = ['num_brigades', 'num_aircraft', 'num_machines',
                   'num_periods', 'seed']

    if columns is None:
        columns = [{'name': 'obj_fun', 'sense': 'min'},
                   {'name': 'solve_time', 'sense': 'min'},
                   {'name': 'elapsed_time', 'sense': 'min'}]

    df = get_best(df, groupby=groupby, columns=columns,
                  best_prefix=best_prefix, rel_prefix=rel_prefix)

    num_instances = sum(df[scatter_by] == list(df[scatter_by])[0])
    performance = dict()

    ref_df = df[df[scatter_by] == reference]
    for m in set(df[scatter_by]).difference({reference}):
        performance[m] = dict()
        print("Scatter: {}".format(m))
        m_df = df[df[scatter_by] == m]
        performance[m] = (ref_df[x].reset_index() - m_df[x].reset_index()).to_dict()[x]

    data = []

    for c in performance.keys():
        data.append(
            go.Scatter(x=list(performance[c].keys()),
                       y=list(performance[c].values()),
                       name=c))

    plotly.offline.iplot(data)


def sim_boxplot(df, y='solve_time', modes=None, columns=None):
    if modes is None:
        modes = ['original', 'fix_work']
    if columns is None:
        columns = ['num_brigades', 'num_aircraft', 'num_machines',
                   'num_periods']

    data = []

    for n, m in enumerate(modes):
        x = ["-".join([str(j) for j in df.loc[[i], columns].values[0]])
             for i in df[df['mode'] == m].index.tolist()]

        data.append(
            go.Box(
                x=x,
                y=df[y][df['mode'] == m],
                legendgroup=m, name=m
            )
        )

    plotly.offline.iplot(data)