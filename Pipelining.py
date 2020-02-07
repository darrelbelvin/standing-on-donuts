checkpoint_file = "./training/checkpoint1.pkl"
n_calls = 1500

from os import path
resume = path.exists(checkpoint_file)

#from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt.utils import use_named_args
from skopt import load

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

import numpy as np

import sys
sys.path.append('..')
from donut_corners.donut_corners import DonutCorners

X, y = load_digits(10, True)
#X, y = X[:100], y[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y)

def on_step(optim_result):
    score = optim_result.fun
    print("best score: %s" % score)

kwargs = {'angle_count': 12 * 1, # must be multiple of 4
            'beam_width': 1.5,
            'fork_spread': 1,
            'beam_length': 4,
            'beam_start': 0.5,
            'beam_round': True,
            'engineered_only': True,
            'grid_size': 4,
            'search_args': dict(img_shape=(8,8), min_grid=0.1, top_n=2, ),
            'eval_method': {'elimination_width': 2, 'max_n': 2, 'elim_double_ends': True},
            }

pipe = Pipeline([
    ('corners', DonutCorners(**kwargs)),
    ('model', SVC(probability=True))
])

# explicit dimension classes can be specified like this
space = [
    Real(0, 4, prior='uniform', name='corners__fork_spread'),
    Real(1.5, 5, prior='uniform', name='corners__beam_width'),
    Real(4, 8, prior='uniform', name='corners__beam_length'),
    Real(0,2, name='corners__beam_start'),
    Categorical([12, 16, 24], name='corners__angle_count')
    #'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    #'model__degree': Integer(1,8),
    #'model__kernel': Categorical(['linear', 'poly', 'rbf']),
]

#space  = [Integer(1, 5, name='max_depth'),
#          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
#          Integer(1, n_features, name='max_features'),
#          Integer(2, 100, name='min_samples_split'),
#          Integer(1, 100, name='min_samples_leaf')]

@use_named_args(space)
def objective(**params):
    pipe.set_params(**params)
    return -np.mean(cross_val_score(pipe, X, y, cv=3, n_jobs=-1,
                                    scoring="neg_log_loss"))

checkpoint_saver = CheckpointSaver(checkpoint_file, compress=9)

if resume:
    res = load(checkpoint_file)
    x0 = res.x_iters
    y0 = res.func_vals
    res_gp = gp_minimize(objective, space, n_calls=n_calls, callback = [on_step, checkpoint_saver], x0=x0, y0=y0)
else:
    res_gp = gp_minimize(objective, space, n_calls=n_calls, callback = [on_step, checkpoint_saver])

print("-------------------")
print("val. score: %s" % res_gp.fun)
print(f"""Best parameters:
- corners__fork_spread = {res_gp.x[0]}
- corners__beam_width = {res_gp.x[1]}
- corners__beam_length = {res_gp.x[2]}
- corners__beam_start = {res_gp.x[3]}
- corners__angle_count = {res_gp.x[4]}
""")

#from skopt.plots import plot_convergence
#plot_convergence(res_gp)