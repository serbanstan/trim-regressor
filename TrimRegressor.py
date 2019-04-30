'''
    TRIM deconfounding algorithm built on top of sklearn-wrap's lasso regression. 
    Link to TRIM paper https://arxiv.org/pdf/1811.05352.pdf
'''


from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# local config
import config

# Custom import commands if any
from sklearn.linear_model.coordinate_descent import Lasso


from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.primitive_interfaces.base import CallResult, DockerContainer
import common_primitives.utils as common_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(params.Params):
    coef_: Optional[ndarray]
    intercept_: Optional[Union[float, ndarray]]
    n_iter_: Optional[int]
    dual_gap_: Optional[float]
    l1_ratio: Optional[float]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    trim_perc = hyperparams.Bounded[float](
        default=.5,
        lower=0,
        upper=1,
        description='The percentage of singular values to be kept unchanged',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    alpha = hyperparams.Bounded[float](
        default=1,
        lower=0,
        upper=None,
        description='Constant that multiplies the L1 term. Defaults to 1.0. ``alpha = 0`` is equivalent to an ordinary least square, solved by the :class:`LinearRegression` object. For numerical reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised. Given this, you should use the :class:`LinearRegression` object.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    # fit_intercept = hyperparams.UniformBool(
    #     default=True,
    #     description='whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    # normalize = hyperparams.UniformBool(
    #     default=False,
    #     description='This parameter is ignored when ``fit_intercept`` is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on an estimator with ``normalize=False``.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    # precompute = hyperparams.Union(
    #     configuration=OrderedDict({
    #         'bool': hyperparams.UniformBool(
    #             default=False,
    #             semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    #         ),
    #         'auto': hyperparams.Constant(
    #             default='auto',
    #             semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    #         )
    #     }),
    #     default='bool',
    #     description='Whether to use a precomputed Gram matrix to speed up calculations. If set to ``\'auto\'`` let us decide. The Gram matrix can also be passed as argument. For sparse input this option is always ``True`` to preserve sparsity.  copy_X : boolean, optional, default True If ``True``, X will be copied; else, it may be overwritten.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    # max_iter = hyperparams.Bounded[int](
    #     default=1000,
    #     lower=0,
    #     upper=None,
    #     description='The maximum number of iterations',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    # tol = hyperparams.Bounded[float](
    #     default=0.0001,
    #     lower=0,
    #     upper=None,
    #     description='The tolerance for the optimization: if the updates are smaller than ``tol``, the optimization code checks the dual gap for optimality and continues until it is smaller than ``tol``.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    # warm_start = hyperparams.UniformBool(
    #     default=False,
    #     description='When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    # positive = hyperparams.UniformBool(
    #     default=False,
    #     description='When set to ``True``, forces the coefficients to be positive.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    # selection = hyperparams.Enumeration[str](
    #     default='cyclic',
    #     values=['cyclic', 'random'],
    #     description='If set to \'random\', a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to \'random\') often leads to significantly faster convergence especially when tol is higher than 1e-4.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )
    
    use_input_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training input. If any specified column cannot be parsed, it is skipped.",
    )
    use_output_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training target. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_input_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not use as training inputs. Applicable only if \"use_columns\" is not provided.",
    )
    exclude_output_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not use as training target. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='new',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )



class TrimRegressor(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive using Trim in combination with Lasso. Code based on JPL's implementation of Lasso. 
    Trim deconfounding paper: https://arxiv.org/pdf/1811.05352.pdf
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_
    """
    
    __author__ = "ISI"
    metadata = metadata_base.PrimitiveMetadata({ 
        # TODO: correct this

        "id": "de250522-5edb-4697-8945-56d04baba0e4",
        "version": "1.0.0",
        "name": "TrimRegressor",
        "description": "Lasso enhanced by spectral deconfounding",
        "python_path": "d3m.primitives.regression.trim_regressor.TrimRegressor",
        "source": {
            "name": "ISI",
            "contact": "mailto:sstan@usc.edu",
            "uris": [ "https://github.com/serbanstan/trim-regressor" ]
        },
        "algorithm_types": ["REGULARIZED_LEAST_SQUARES", 'FEATURE_SCALING'],
        "primitive_family": "FEATURE_CONSTRUCTION",
        "installation": [ config.INSTALLATION ]

         # "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LASSO, ],
         # "name": "sklearn.linear_model.coordinate_descent.Lasso",
         # "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
         # "python_path": "d3m.primitives.regression.lasso.SKlearn",
         # "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html']},
         # "version": "v2019.2.27",
         # "id": "a7100c7d-8d8e-3f2a-a0ee-b4380383ed6c",
         # 'installation': [
         #                # TODO : Will update based on https://gitlab.com/datadrivendiscovery/d3m/issues/137
         #                #{
         #                #    "type": "PIP",
         #                #    "package_uri": "git+https://gitlab.com/datadrivendiscovery/common-primitives.git@26419dde2f660f901066c896a972ae4c438ee236#egg=common_primitives"
         #                #},
         #                {'type': metadata_base.PrimitiveInstallationType.PIP,
         #                   'package_uri': 'git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@{git_commit}#egg=sklearn_wrap'.format(
         #                       git_commit=utils.current_git_commit(os.path.dirname(__file__)),
         #                    ),
         #                   }]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = Lasso(
              alpha=self.hyperparams['alpha'],
              # fit_intercept=self.hyperparams['fit_intercept'],
              # normalize=self.hyperparams['normalize'],
              # precompute=self.hyperparams['precompute'],
              # max_iter=self.hyperparams['max_iter'],
              # tol=self.hyperparams['tol'],
              # warm_start=self.hyperparams['warm_start'],
              # positive=self.hyperparams['positive'],
              # selection=self.hyperparams['selection'],
              random_state=self.random_seed,
        )
        # self._F = None
        # self._F_inv = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._fitted = False
        
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._training_outputs, self._target_names, self._target_column_indices = self._get_targets(outputs, self.hyperparams)
        self._fitted = False

    # Computes the linear transform F, so we work in the system (FX, FY) to recover the
    # true betas. 
    def _compute_F(self, X_data):
        X = numpy.array(X_data)

        U,d,V = numpy.linalg.svd(X)

        r = len(d)

        tau = sorted(d)[int(r * self.hyperparams['trim_perc'])]

        d_hat = numpy.array([min(x, tau)/x for x in d])

        D_hat = numpy.zeros(U.shape)
        D_hat[:r, :r] = numpy.diag(d_hat)

        D_hat_inv = numpy.zeros(U.shape)
        D_hat_inv[:r, :r] = numpy.diag(1 / d_hat)

        F = numpy.dot(U, numpy.dot(D_hat, U.T))
        F_inv = numpy.dot(U, numpy.dot(D_hat_inv, U.T))

        return F, F_inv
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")
        self._target_columns_metadata = self._get_target_columns_metadata(self._training_outputs.metadata)
        sk_training_output = self._training_outputs.values

        shape = sk_training_output.shape
        if len(shape) == 2 and shape[1] == 1:
            sk_training_output = numpy.ravel(sk_training_output)

        # print(self._training_inputs)

        F, _ = self._compute_F(self._training_inputs)

        new_inputs = numpy.dot(F, self._training_inputs)
        new_outputs = numpy.dot(F, sk_training_output)

        self._clf.fit(new_inputs, new_outputs)
        self._beta = self._clf.coef_

        remainder = sk_training_output - np.dot(self._training_inputs, beta)
        self._delta = self._clf.fit(self._training_inputs, remainder)

        self._fitted = True

        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]

        sk_output = numpy.dot(sk_inputs, self._beta + self._delta)

        # F, F_inv = self._compute_F(sk_inputs)

        # new_inputs = numpy.dot(F, sk_inputs)
        # trim_output = self._clf.predict(new_inputs)
        # sk_output = numpy.dot(F_inv, trim_output)

        if sparse.issparse(sk_output):
            sk_output = sk_output.toarray()
        output = self._wrap_predictions(inputs, sk_output)
        output.columns = self._target_names
        outputs = common_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                               add_index_columns=self.hyperparams['add_index_columns'],
                                               inputs=inputs, column_indices=self._target_column_indices,
                                               columns_list=[output])

        return CallResult(outputs)
        
    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                beta=None,
                delta=None,
                # coef_=None,
                # intercept_=None,
                # n_iter_=None,
                # dual_gap_=None,
                # l1_ratio=None,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            beta=self._beta,
            delta=self._delta,
            # coef_=getattr(self._clf, 'coef_', None),
            # intercept_=getattr(self._clf, 'intercept_', None),
            # n_iter_=getattr(self._clf, 'n_iter_', None),
            # dual_gap_=getattr(self._clf, 'dual_gap_', None),
            # l1_ratio=getattr(self._clf, 'l1_ratio', None),
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_columns_metadata_=self._target_columns_metadata,
            target_column_indices_=self._target_column_indices
        )

    def set_params(self, *, params: Params) -> None:
        self._beta = params['beta'],
        self._delta = params['delta'],
        # self._clf.coef_ = params['coef_']
        # self._clf.intercept_ = params['intercept_']
        # self._clf.n_iter_ = params['n_iter_']
        # self._clf.dual_gap_ = params['dual_gap_']
        # self._clf.l1_ratio = params['l1_ratio']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        self._fitted = False
        
        if params['coef_'] is not None:
            self._fitted = True
        if params['intercept_'] is not None:
            self._fitted = True
        if params['n_iter_'] is not None:
            self._fitted = True
        if params['dual_gap_'] is not None:
            self._fitted = True
        if params['l1_ratio'] is not None:
            self._fitted = True
    


    
    
    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_input_columns'],
                                                                             exclude_columns=hyperparams['exclude_input_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, numpy.integer, numpy.float64)
        accepted_semantic_types = set()
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False
    
    @classmethod
    def _get_targets(cls, data: d3m_dataframe, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return data, list(data.columns), []

        metadata = data.metadata

        def can_produce_column(column_index: int) -> bool:
            accepted_semantic_types = set()
            accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/TrueTarget")
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = set(column_metadata.get('semantic_types', []))
            if len(semantic_types) == 0:
                cls.logger.warning("No semantic types found in column metadata")
                return False
            # Making sure all accepted_semantic_types are available in semantic_types
            if len(accepted_semantic_types - semantic_types) == 0:
                return True
            return False

        target_column_indices, target_columns_not_to_produce = common_utils.get_columns_to_use(metadata,
                                                                                               use_columns=hyperparams[
                                                                                                   'use_output_columns'],
                                                                                               exclude_columns=
                                                                                               hyperparams[
                                                                                                   'exclude_output_columns'],
                                                                                               can_use_column=can_produce_column)
        targets = common_utils.select_columns(data, target_column_indices)
        target_column_names = []
        for idx in target_column_indices:
            target_column_names.append(data.columns[idx])
        return targets, target_column_names, target_column_indices

    @classmethod
    def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata) -> List[OrderedDict]:
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = list(column_metadata.get('semantic_types', []))
            semantic_types_to_remove = ["https://metadata.datadrivendiscovery.org/types/TrueTarget",
                                        "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                                        "https://metadata.datadrivendiscovery.org/types/Attribute"]
            if 'https://metadata.datadrivendiscovery.org/types/PredictedTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            semantic_types = [semantic_type for semantic_type in semantic_types if
                              semantic_type not in semantic_types_to_remove]
            column_metadata['semantic_types'] = semantic_types

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata
    
    @classmethod
    def _update_predictions_metadata(cls, inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
                                     target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:
        outputs_metadata = inputs_metadata.clear(for_value=outputs, generate_metadata=True)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:
        outputs = d3m_dataframe(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, self._target_columns_metadata)
        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata):
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict()
            semantic_types = []
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            column_name = outputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
            if column_name is None:
                column_name = "output_{}".format(column_index)
            column_metadata["semantic_types"] = semantic_types
            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


# TrimRegressor.__doc__ = Lasso.__doc__