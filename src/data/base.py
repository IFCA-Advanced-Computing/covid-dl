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

from paths import PATHS


def get_data(trend=False,
             shift_window=None,
             labels=True,
             multistep=None,
             exclude=['Ceuta', 'Melilla', 'Balears, Illes',
                      'Santa Cruz de Tenerife', 'Palmas, Las']):
    """
    trend : bool
        Compute velocities and accelerations
    shift_window : list of ints
        If trend=False, use raw past days.
    labels : bool
        Compute the labels
    multistep : int, None
        If labels=True, then compute all shifted days till this date.
        If set to None, only compute label t+7.
    exclude : list of strs
        Outlier provinces to exclude from the dataset
    """

    df = pd.read_csv(
        PATHS.rawdir / "provinces-incidence-mobility.csv",
        header=[0, 1]
    )
    df = df.fillna(0)

    # Format column names correctly
    for i, col in enumerate(df.columns.levels):
        new = np.where(col.str.contains('Unnamed'), '', col)
        df = df.rename(columns=dict(zip(col, new)), level=i)

    # Load density/population data
    prov_path = PATHS.rawdir / 'prov_data.xls'
    prov_data = pd.read_excel(prov_path, dtype={'Provincia': 'string'})
    den_map = dict(zip(prov_data.Provincia, prov_data.Densidad))
    pop_map = dict(zip(prov_data.Provincia, prov_data.Poblacion))

    # Process provinces
    new_df = []
    provinces = df[('province', '')].unique()
    for p in provinces:

        if p in exclude:
            continue

        dfp = df[df[('province', '')] == p]

        new_dfp = dfp[['date', 'province', 'incidence 7', 'flux intra']]
        new_dfp = new_dfp.droplevel(1, axis='columns')

        # Compute a 7-day rolling sum for mobility
        cols = dfp.xs('flux', level=1, drop_level=False, axis='columns').columns
        dfp[cols] = dfp[cols].rolling(7).sum()
        new_dfp['flux intra'] = new_dfp['flux intra'].rolling(7).sum()

        # Add external risk
        flux = dfp.xs('flux', level=1, drop_level=True, axis='columns')
        inc = dfp.xs('incidence 7', level=1, drop_level=True, axis='columns')
        new_dfp['external risk'] = (flux * inc).sum(axis='columns', min_count=1)

        # Add province density
        # new_dfp['density'] = den_map[p]

        # Normalize external risk and flux_intra by population
        # new_dfp['external risk'] = new_dfp['external risk'] / pop_map[p]
        # new_dfp['flux intra'] = new_dfp['flux intra'] / pop_map[p]

        # Apply log for normalizing scales
        # if log:
        #     new_dfp['external risk'] = np.log(new_dfp['external risk'])
            # new_dfp['density'] = np.log(new_dfp['density'])
            # new_dfp['incidence 7'] = np.log(new_dfp['incidence 7'] + 1)  # +1 to avoid log(0)

        # Add time window as additional columns
        shift_cols = ['incidence 7', 'flux intra', 'external risk']
        shift_days = []
        if trend:
            shift_days = [-1, -3]
        elif shift_window:
            shift_days = range(shift_window, 0)
        for i in shift_days:
            for j in shift_cols:
                new_dfp[f'{j} (t{i:+d})'] = new_dfp[j].shift(-i)

        # Create a shifted incidence for label
        if labels:
            if multistep:
                for i in range(1, multistep+1):
                    new_dfp[f'incidence 7 (t+{i})'] = new_dfp['incidence 7'].shift(-i)
            else:
                new_dfp['incidence 7 (t+7)'] = new_dfp['incidence 7'].shift(-7)

        # Keep only rows with non-nans (eg. t+7 shift will remove last 7 rows)
        new_dfp = new_dfp[~new_dfp.isna().any(axis='columns')]

        # Compute trends (works best normalizing with value at t)
        if trend:
            for i in shift_cols:
                d1, d2 = shift_days[0], shift_days[1]
                new_dfp[f'{i} (vel)'] = (new_dfp[f'{i}'] - new_dfp[f'{i} (t{d1})']) \
                                       / new_dfp[f'{i}']
                new_dfp[f'{i} (acc)'] = (new_dfp[f'{i}'] - 2 * new_dfp[f'{i} (t{d1})'] + new_dfp[f'{i} (t{d2})']) \
                                       / new_dfp[f'{i}']
                new_dfp = new_dfp.replace([np.inf, -np.inf, np.nan], 0)
                new_dfp = new_dfp.drop(columns=[f'{i} (t{d1})', f'{i} (t{d2})'])

        # Append province to final df
        new_df.append(new_dfp)

    new_df = pd.concat(new_df).reset_index(drop=True)
    new_df = new_df.sort_index(axis=1)

    return new_df


def make_splits(df,
                multistep=None,
                norm=True,
                dtrain='2020-11-30',  # end of training dataset
                dval='2021-01-31',  # end of validation dataset
                ):

    # Create X,y
    df = df.set_index(['date', 'province'])
    if multistep:
        labels = [f'incidence 7 (t+{i})' for i in range(1, multistep+1)]
    else:
        labels = ['incidence 7 (t+7)']
    X = df[df.columns.difference(labels)]
    y = df[labels]

    # # Label is difference of inc, instead of raw inc
    # inc = X.pop('incidence 7')
    # y['incidence 7 (t+7)'] = y['incidence 7 (t+7)'] - inc

    # Split train/val/test
    dates = df.index.get_level_values('date')
    if not dtrain:
        dtrain = max(dates)
    if not dval:
        dval = max(dates)
    assert dval >= dtrain, 'Validation date should be bigger than training date.'

    splits = {'train': {}, 'val': {}, 'test': {}}
    splits['train']['X'] = X[dates <= dtrain]
    splits['train']['y'] = y[dates <= dtrain]
    splits['val']['X'] = X[(dates > dtrain) & (dates <= dval)]
    splits['val']['y'] = y[(dates > dtrain) & (dates <= dval)]
    splits['test']['X'] = X[dates > dval]
    splits['test']['y'] = y[dates > dval]

    # Normalize inputs
    if norm:
        mean = splits['train']['X'].mean()
        std = splits['train']['X'].std()

        for i in ['train', 'val', 'test']:
            splits[i]['X'] = (splits[i]['X'] - mean) / std

        # Save to df
        norm = pd.DataFrame([mean, std], index=['mean', 'std'])
        norm.to_csv(PATHS.models / 'norm.csv')

    return splits


def normalize(X):
    norm = pd.read_csv(PATHS.models / 'norm.csv', index_col=0)
    return (X - norm.loc['mean']) / norm.loc['std']


def unprocess(df, norm=True, log=True):
    """
    Undo the data preprocessing
    """
    normdf = pd.read_csv(PATHS.models / 'norm.csv', index_col=0)
    df = df.copy()  # avoid weird object id issue (function did not create a new df)
    for c in df.columns:
        cn = c.split(' (')[0]  # 'incidence 14 (t+14)' --> 'incidence 14'

        if norm:
            df[c] = df[c] * normdf.at['std', cn] + normdf.at['mean', cn]

        if log and (cn in ['external risk']):
            df[c] = np.exp(df[c])

        # if log and (cn in ['external risk', 'density', 'incidence 14']):
        #     df[c] = np.exp(df[c])
        #     if cn == 'incidence 14':
        #         df[c] -= 1

        if cn == 'incidence 14':
            df[c] = np.round(df[c]).astype(np.int)

    return df


def single_X_y(splits):
    X, y = [], []
    for i in ['train', 'val', 'test']:
        X.append(splits[i]['X'])
        y.append(splits[i]['y'])
    X = pd.concat(X)
    y = pd.concat(y)
    return X, y
