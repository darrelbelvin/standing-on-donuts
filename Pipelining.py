#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append('..')
from donut_corners.donut_corners import DonutCorners


# In[4]:


X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

def on_step(optim_result):
    score = opt.best_score_
    print("best score: %s" % score)
    if score >= 0.99:
        print('Interrupting!')
        return True


# In[6]:


# pipeline class is used as estimator to enable
# search over different model types
kwargs = {'angle_count': 12 * 1, # must be multiple of 4
            'beam_count': 12 * 1,
            'beam_width': 2,
            'fork_spread': 1,
            'beam_length': 3,
            'beam_start': 0,
            'beam_round': True,
            'search_args': dict(img_shape=(8,8), edge_offset = 1, top_n = 2, max_rounds = 10, max_step=4, initial_simplex_size=3),
            'eval_method': {'sectional': False, 'elimination_width': 2, 'max_n': 2, 'elim_double_ends': True},
            }

pipe = Pipeline([
    ('corners', DonutCorners(**kwargs)),
    ('model', SVC())
])

# explicit dimension classes can be specified like this
svc_search = {
    'corners__fork_spread': Integer(0,2),
    'corners__beam_length': Integer(2,8),
    'corners__beam_start': Integer(0,3),
    #'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    #'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    #'model__degree': Integer(1,8),
    #'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(
    pipe,
    svc_search,
    n_iter=2,
    cv=3
)

opt.fit(X_train, y_train, callback=on_step)

print("-------------------")
print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))


# In[ ]:




