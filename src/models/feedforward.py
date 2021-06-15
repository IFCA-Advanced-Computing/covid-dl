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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import numpy as np
import pandas as pd


def bayesian_predict(model, X, samples=10):
    """
    Modified predict function to account for bayesian uncertainty of Dropout
    """
    y = []
    X = np.array(X)
    for _ in range(samples):
        ysample = np.array(model(X,
                                 training=True))  # activate dropout during inference
        y.append(ysample)
    y = np.array(y)
    return y.mean(axis=0), y.std(axis=0)


def model_summary(model, splits):

    print('\n# Metrics')
    metrics = {'mae': {}, 'mape': {}}

    for i in ['train', 'val', 'test']:
        x_t = splits[i]['X']
        y_t = splits[i]['y']

        if len(x_t) != 0:
            _, y_err = bayesian_predict(model, x_t)
            y_p = model.predict(x_t)  # normal inference (no dropout)

            # We just compare results for t+7 to be able to compare with single step
            y_t = pd.Series(y_t[:, -1])
            y_p = pd.Series(y_p[:, -1])

            mae = np.abs(y_p - y_t)
            mape = np.abs((y_p - y_t) / y_t)
            mape = mape.replace([np.inf, -np.inf], np.nan)

            metrics['mae'][i] = mae.mean()
            metrics['mape'][i] = mape.mean()

            print(f' - {i.capitalize()}')
            print(f'   Mean Absolute Error (in incidence): {mae.mean():.4f}')
            print(f'   Mean Absolute Percentage Error (in incidence): {mape.mean():.4f}')

    return metrics


def build_model(layer_num, layer_cells):

    model = Sequential()
    for i in range(layer_num):
        model.add(Dense(layer_cells,
                        activation="relu",
                        kernel_initializer="he_uniform",
                        name=f"dense{i}"))
        model.add(Dropout(0.05))

    model.add(Dense(7,
                    activation="relu",
                    name="output"))
    return model
