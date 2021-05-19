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

import pathlib
import os
import shutil


class Paths:
    def __init__(self):
        b = pathlib.Path(__file__).resolve()
        self._base = b.parents[1]

    @property
    def base(self):
        return self._base

    @property
    def models(self):
        return self.base / 'models'

    @property
    def data(self):
        return self.base / 'data'

    @property
    def rawdir(self):
        return self.data / 'raw'

    @property
    def outdir(self):
        return self.data / 'predictions'

    def __str__(self):
        return f"""
                 Base path: {self.base}
                 Data path: {self.data}
                 Models path: {self.models}
    """


PATHS = Paths()


def check_dirs(regenerate=False):
    print("Checking directories...")
    print(PATHS)
    PATHS.base.exists() or os.makedirs(PATHS.base)

    if PATHS.outdir.exists() and not regenerate:
        print(f"\t'{PATHS.outdir}' exists, not overwriting it")
        return False
    elif PATHS.outdir.exists() and regenerate:
        print(f"\t'{PATHS.outdir}' exists, deleting it")
        shutil.rmtree(PATHS.outdir, ignore_errors=True)
    return True
