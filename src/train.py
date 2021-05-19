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
import tensorflow.keras as keras

from paths import PATHS
from data.base import make_splits, get_data, single_X_y
from models.feedforward import build_model, model_summary, bayesian_predict


# Load data
df = get_data(trend=True, multistep=7)
splits = make_splits(df,
                     norm=True,
                     multistep=7,
                     dtrain=None,  # all is training
                     dval=None)
X, y = single_X_y(splits)

# Sample weights (recent samples have higher weight)
dates = splits['train']['X'].index.get_level_values(0)
dates = pd.to_datetime(dates)
n = (dates[-1] - dates).days
w = 0.99 ** n  # geometric discount (works better than linear)

# Formatting for this specific notebook: Transform to arrays of float32
for i in ['train', 'val', 'test']:
    splits[i]['X'] = splits[i]['X'].values.astype(np.float32)
    splits[i]['y'] = splits[i]['y'].values.astype(np.float32)

# Train
keras.backend.clear_session()

model = build_model(layer_num=8, layer_cells=500)

model.compile(loss='MeanSquaredError',
              optimizer=keras.optimizers.Adam(amsgrad=True),
              metrics=['MeanAbsoluteError'])

history = model.fit(x=splits['train']['X'],
                    y=splits['train']['y'],
                    validation_data=(splits['val']['X'],
                                     splits['val']['y']),
                    sample_weight=w,
                    batch_size=12000,
                    epochs=80,
                    verbose=0,
                    shuffle=True)

print(f"Trained for {len(history.history['loss'])} epochs")

metrics = model_summary(model, splits)

_, y_err = bayesian_predict(model, X)
y_pred = model.predict(X)  # normal inference (no dropout)
y_pred = np.round(y_pred).astype(np.int)

# Rescale errors according to 95% confidence interval
alpha_df = np.abs((y - y_pred) / (2 * y_err))
alpha = alpha_df.quantile(0.95, axis=0)

# Save everything
mpath = PATHS.models / 'feedforward'
mpath.mkdir(parents=True, exist_ok=True)

tmp = pd.DataFrame.from_dict(metrics, orient='index')
tmp.to_csv(mpath / 'metrics.csv')

tmp = pd.DataFrame.from_dict(history.history, orient='columns')
tmp.to_csv(mpath / 'history.csv', index=False)

alpha.to_csv(mpath / 'error_scales.csv')
model.save(mpath / 'model')
