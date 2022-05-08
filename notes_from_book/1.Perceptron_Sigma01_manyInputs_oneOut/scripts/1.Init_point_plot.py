import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix

import saveFigure

# initiating random number
np.random.seed(11)

#### Creating the dataset

# mean and standard deviation for the x belonging to the first class
mu_x1, sigma_x1 = 0, 0.1

# constat to make the second distribution different from the first
x2_mu_diff = 0.35

# creating the first distribution
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'type': 0})

# creating the second distribution
d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'type': 1})

data = pd.concat([d1, d2], ignore_index=True)
ax = sns.scatterplot(x="x1", y="x2", hue="type",data=data)

saveFigure.save ( ax , '1.first_plot_distribution' )
