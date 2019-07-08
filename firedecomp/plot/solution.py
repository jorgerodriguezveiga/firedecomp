"""Module to represent the solutions."""
# Python packages

import numpy as np
import plotly.graph_objs as go
import plotly
# plotly.offline.init_notebook_mode()


# plot_contention -------------------------------------------------------------
def plot_contention(problem):
    min_per = min(problem.wildfire.get_names()) - 1
    contained_dict = problem.wildfire.get_info("contained")
    perimeter_dict = problem.wildfire.get_info("increment_perimeter")
    contained_perimeter_dict = {t: (not contained_dict[t])*perimeter_dict[t]
                                for t in problem.wildfire.get_names()}

    # Create traces
    y_peri = list(np.cumsum([contained_perimeter_dict[k]
                             for k in contained_perimeter_dict]))
    perimeter = go.Scatter(
        x=[min_per] + [k for k in contained_perimeter_dict],
        y=[y_peri[0]] + y_peri,
        mode='lines+markers',
        name='perimeter',
        line=dict(
            shape='vh'
        )
    )

    working_dict = problem.resources_wildfire.get_info("work")
    resource_performance_dict = problem.resources_wildfire.get_info(
        "resource_performance")
    working_performance_dict = {t: sum(
        [working_dict[i, t] * resource_performance_dict[i, t] for i in
         problem.resources.get_names()]) for t in problem.wildfire.get_names()}

    y_perf = list(np.cumsum(
        [working_performance_dict[k] for k in working_performance_dict]))
    performance = go.Scatter(
        x=[min_per] + [k for k in working_performance_dict],
        y=[y_perf[0]] + y_perf,
        mode='lines+markers',
        name='performance',
        line=dict(
            shape='vh'
        )
    )

    data = [perimeter, performance]

    layout = dict(title='Wildfire contention',
                  xaxis=dict(title='Periods'),
                  yaxis=dict(title='Perimeter (km)'),
                  )
    plotly.offline.iplot({'data': data, 'layout': layout},
                         filename='scatter-mode')
# --------------------------------------------------------------------------- #


# plot_scheduling -------------------------------------------------------------
def plot_scheduling(problem):
    def get_resources_wildfire_scatters(problem, info):
        if info == 'work':
            line = dict(
                color='rgb(200, 0, 0)'
            )
        elif info == 'rest':
            line = dict(
                color='rgb(0, 200, 0)',
                dash='dash')
        elif info == 'travel':
            line = dict(
                color='rgb(0, 0, 200)',
                dash='dot')
        else:
            raise ValueError('Unknown info value.')

        res_names = problem.resources.get_names()
        per_names = problem.wildfire.get_names()
        min_per = min(per_names) - 1
        max_per = max(per_names) + 1
        info_dict = problem.resources_wildfire.get_info(info)
        info_dict.update({(i, min_per): False for i in res_names})
        info_dict.update({(i, max_per): False for i in res_names})
        scatter_dict = {
            i: [p
                if info_dict[i, p] or info_dict[i, p + 1] else
                None
                for p in [0] + per_names]
            for i in res_names}
        first_legend_info = min(
            [i for i, p in scatter_dict.items() if
             not all(v is None for v in p)])
        return [
            go.Scatter(
                x=p,
                y=[i] * len(p),
                mode='lines+markers',
                name=info,
                legendgroup=info,
                line=line)
            if i == first_legend_info else
            go.Scatter(
                x=p,
                y=[i] * len(p),
                mode='lines+markers',
                name=info,
                legendgroup=info,
                showlegend=False,
                line=line)
            for t, (i, p) in enumerate(scatter_dict.items()) if
            not all(v is None for v in p)]

    data = []
    data += get_resources_wildfire_scatters(problem, 'rest')
    data += get_resources_wildfire_scatters(problem, 'travel')
    data += get_resources_wildfire_scatters(problem, 'work')

    layout = dict(title='Resources Scheduling',
                  xaxis=dict(title='Periods'),
                  yaxis=dict(title='Resources'),
                  showlegend=True
                  )
    plotly.offline.iplot({'data': data, 'layout': layout},
                         filename='scatter-mode')
# --------------------------------------------------------------------------- #
