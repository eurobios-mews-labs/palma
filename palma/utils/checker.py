# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from sklearn.utils.validation import check_X_y

from palma import Project


_CLASSIFICATION_METRICS = [
    "accuracy", "balanced_accuracy", "top_k_accuracy", "average_precision",
    "neg_brier_score", "f1", "f1_micro", "f1_macro", "f1_weighted",
    "f1_sample", "neg_log_loss", "precision", "precision_macro",
    "precision_micro", "precision_samples", "precision_weighted", "recall",
    "recall_macro", "recall_micro", "recall_samples", "recall_weighted",
    "jaccard", "jaccard_macro", "jaccard_micro", "jaccard_samples",
    "jaccard_weighted", "roc_auc", "roc_auc_ovr", "roc_auc_ovo",
    "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"
]

_REGRESSION_METRICS = [
    "explained_variance", "r2", "max_error", "neg_median_absolute_error",
    "neg_mean_absolute_error", "neg_mean_absolute_percentage_error",
    "neg_mean_squared_error", "neg_mean_squared_log_error",
    "neg_root_mean_squared_error", "neg_mean_poisson_deviance",
    "neg_mean_gamma_deviance"
]


class ProjectPlanChecker(object):
    """
    ProjectPlanChecker is an object that checks the project plan.

    At the :meth:`~palma.project.Project.build` moment, this object \
    run several checks in order to see if the project plan is well designed.

    Here is an overview of the checks performed by the object:
        - :meth:`~palma.utils.checker.ProjectPlanChecker._check_arrays`\
        : see whether X and y attribute are compliant with \
        sklearn standards.
        - :meth:`~palma.utils.checker.ProjectPlanChecker._check_project_problem`: see if the problem type is correctly \
        informed by the user.
        - :meth:`~palma.utils.checker.ProjectPlanChecker._check_problem_metrics`: see if the known metrics are consistent with \
        the project problem
    """

    def _check_arrays(self, project: Project) -> None:
        _, _ = check_X_y(project.X,
                         project.y,
                         dtype=None,
                         force_all_finite='allow-nan')

    def _check_project_problem(self, project: Project) -> None:
        if not project.problem in ["classification", "regression"]:
            raise ValueError(
                f"Unknown problem: {project.problem}, please see documentation"
            )

    def run_checks(self, project: Project) -> None:
        """
        Perform some tests on the project plan

        Several checks are performed in order to check if the
        project plan is consistent:
            - checks the project problem 
            - checks the metrics provided by the user
            - checks the data provided by the user (scikit learn wrapper)

        Parameters
        ----------
        project : :class:`~autolm.project.Project`
            an Project instance
        """
        self._check_project_problem(project)
        self._check_arrays(project)
