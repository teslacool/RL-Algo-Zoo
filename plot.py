import argparse
import os
import os.path as osp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DIV_LINE_WIDTH = 50
exp_idx = 0
units = dict()

def get_datasets(logdir, condition=None, **kwargs):
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.csv' in files:
            condition1 = condition
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1
            try:
                exp_data = pd.read_table(osp.join(root, 'progress.csv'), sep=',')
            except Exception as e:
                print('Could not read from %s' % os.path.join(root, 'progress.csv'))
                print(e)
                exit(1)
            performance = kwargs.get('yaxis', 'eprewmean')
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets



def get_all_dataset(all_logdirs, legend=None, select=None, exclude=None, **kwargs):
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.osp:
            logdirs.append(logdir)
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs.extend([fulldir(x) for x in listdir if prefix in x])
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(x not in log for x in exclude)]

    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)
    if legend is not None:
        if len(legend) != len(logdirs):
            print('\nlegend...\n' + '=' * DIV_LINE_WIDTH + '\n')
            for l in legend:
                print(l)
            print('\n' + '=' * DIV_LINE_WIDTH)
            print("Must give a legend title for each set of experiments.")
            exit(1)
    else:
        legend = []
        for logdir in logdirs:
            legend.append(logdir.rstrip(os.sep).split(os.sep)[-1])

    data = []
    for log, leg in zip(logdirs, legend):
        print('\nloading data from {}  /  {}\n'.format(log, leg))
        data.extend(get_datasets(log, leg, **kwargs))
    return data


def plot_data(data, xaxis='timesteps', value="Performance", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    fig = plt.figure(kwargs['figname'], figsize=(8, 6))
    sns.set(style="darkgrid", font_scale=1.8)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd',)
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(loc='best').set_draggable(True)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

    figdir = 'data/figs'
    os.makedirs(figdir, exist_ok=True)
    figname = osp.join(figdir, '{}.pdf'.format(kwargs['figname']))
    fig.savefig(figname, format='pdf')
    plt.show()


def make_plots(logdir, legend, xaxis, yaxis, xlabel, ylabel,
               count, smooth, select, exclude, est, **kwargs):
    data = get_all_dataset(logdir, legend, select, exclude,
                    xaxis=xaxis, yaxis=yaxis)
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, est)
    xlabel = xlabel if xlabel else xaxis
    ylabel = ylabel if ylabel else yaxis
    plot_data(data, xaxis=xaxis, value='Performance', condition=condition, smooth=smooth,
              estimator=estimator, xlabel=xlabel, ylabel=ylabel, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str, nargs='+')
    parser.add_argument('--legend', type=str, nargs='*', default=None)
    parser.add_argument('--xaxis', '-x', type=str, default='timesteps',)
    parser.add_argument('--yaxis', '-y', type=str, default='eprewmean',)
    parser.add_argument('--xlabel', type=str, default=None)
    parser.add_argument('--ylabel', type=str, default=None)
    parser.add_argument('--figname', type=str, default='fig')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*', default=None)
    parser.add_argument('--exclude', nargs='*', default=None)
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    make_plots(**vars(args))


if __name__ == "__main__":
    main()