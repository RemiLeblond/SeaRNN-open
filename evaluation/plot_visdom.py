from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import os
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--log_path', default='.', type=str, help="Folder to search for logs")
parser.add_argument('--log_names', default=[], nargs='+', type=str, help="Plot logs from these folder")

opt = parser.parse_args()

if not os.path.isdir(opt.log_path):
    raise (RuntimeError("Log path %s does not exist" % opt.log_path))

viz = Visdom()
viz_plots = {}
x_name_all = ['iter', 'time']


def vizualize_log(log_path):
    # read the log_file
    log_file = os.path.join(opt.log_path, log_path, "train_log.pkl")
    if not os.path.isfile(log_file):
        print('WARNING: Could not find file %s' % log_file)
        return
    logs = pickle.load(open(log_file, "rb"))

    for x_name in x_name_all:
        if x_name in logs:
            for y_name, y_data in logs.items():
                if not y_name in x_name_all:
                    x_data = logs[x_name]
                    plot_key = (y_name, x_name)

                    plot_opts = dict(
                        markers=False,
                        xlabel=x_name,
                        ylabel=y_name,
                        title='{0} vs. {1}'.format(y_name, x_name),
                        legend=[os.path.basename(os.path.normpath(log_path))]
                    )
                    X = np.array(x_data)
                    Y = np.array(y_data)

                    mask_non_nan = np.logical_not(np.isnan(Y))
                    X = X[mask_non_nan]
                    Y = Y[mask_non_nan]

                    if plot_key in viz_plots:
                        viz.updateTrace(
                            X=X,
                            Y=Y,
                            win=viz_plots[plot_key],
                            name=plot_opts['legend'][0],
                            opts=plot_opts
                        )
                    else:
                        viz_plots[plot_key] = viz.line(Y=Y, X=X, opts=plot_opts)

if len(opt.log_names) == 0:
    print('--log_names was not specified, scanning folder %s' % opt.log_path)
    log_names = next(os.walk(opt.log_path))[1]
else:
    log_names = opt.log_names


n = len(log_names)
for i_log, path in enumerate(log_names):
    try:
        vizualize_log(path)
        print('Plot %d of %d: %s' % (i_log, n, path))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
         print("Failed to plot from %s. Error: %s" % (path, str(e)))
