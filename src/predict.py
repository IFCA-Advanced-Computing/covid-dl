# Copyright (c) 2020 Spanish National Research Council
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import tensorflow as tf

from paths import PATHS
from data.base import get_data, normalize
from models.feedforward import bayesian_predict


# Load model
mpath = PATHS.models / 'feedforward'
if not mpath.exists():
    raise Exception('You should train the model before predicting.')
model = tf.keras.models.load_model(mpath / 'model')
alpha = pd.read_csv(mpath / 'error_scales.csv', index_col=0)
alpha = alpha['0.95']

# Load data
df = get_data(trend=True, labels=False)
X = df.set_index(['date', 'province'])
X = normalize(X)

# Keep only last date
dates = X.index.get_level_values('date')
X = X.loc[max(dates)]

# Predict
_, y_err = bayesian_predict(model, X)
y_pred = model.predict(X)  # normal inference (no dropout)
y_pred = np.round(y_pred).astype(np.int)

new_err = y_err * np.array(alpha)[None, :]

y_pred = np.round(y_pred).astype(np.int)
new_err = np.ceil(new_err).astype(np.int)

# Format into pandas dataframe with dates
drange = pd.date_range(start=max(dates), periods=8)[1:]
index = pd.MultiIndex.from_product([drange, X.index],
                                   names=['date', 'province'])
data = np.vstack((y_pred.T.reshape(-1),
                  new_err.T.reshape(-1))).T
pred_df = pd.DataFrame(data,
                       index=index,
                       columns=['incidence 7 (mean)',
                                'incidence 7 (std)'])

pred_df.to_csv(PATHS.outdir / 'predictions.csv')
