# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

from palma.components.base import ProjectComponent

try:
    from pandas_profiling import ProfileReport
except ImportError:
    ProfileReport = None


class ProfilerYData(ProjectComponent):

    def __init__(self,
                 **config,
                 ):
        self.config = config

    def __call__(self, project: "Project"):
        profile = ProfileReport(
            pd.concat((project.X, project.y)),
            **self.config,
            title="Pandas Profiling Report")
        profile.to_file(f"report_{project.project_name}.html")

