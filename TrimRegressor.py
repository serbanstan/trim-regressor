from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.linear_model.logistic import LogisticRegression


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
    intercept_: Optional[ndarray]
    n_iter_: Optional[ndarray]
    classes_: Optional[ndarray]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    penalty = hyperparams.Enumeration[str](
        values=['l1', 'l2'],
        default='l2',
        description='Used to specify the norm used in the penalization. The \'newton-cg\', \'sag\' and \'lbfgs\' solvers support only l2 penalties.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    dual = hyperparams.UniformBool(
        default=False,
        description='Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    fit_intercept = hyperparams.UniformBool(
        default=True,
        description='Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    intercept_scaling = hyperparams.Hyperparameter[float](
        default=1,
        description='Useful only when the solver \'liblinear\' is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a "synthetic" feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes ``intercept_scaling * synthetic_feature_weight``.  Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    class_weight = hyperparams.Union(
        configuration=OrderedDict({
            'str': hyperparams.Constant(
                default='balanced',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Hyperparameter[None](
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Weights associated with classes in the form ``{class_label: weight}``. If not given, all classes are supposed to have weight one.  The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.  Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.  .. versionadded:: 0.17 *class_weight=\'balanced\'* instead of deprecated *class_weight=\'auto\'*.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_iter = hyperparams.Bounded[int](
        default=100,
        lower=0,
        upper=None,
        description='Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    solver = hyperparams.Enumeration[str](
        values=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        default='liblinear',
        description='Algorithm to use in the optimization problem.  - For small datasets, \'liblinear\' is a good choice, whereas \'sag\' is faster for large ones. - For multiclass problems, only \'newton-cg\', \'sag\' and \'lbfgs\' handle multinomial loss; \'liblinear\' is limited to one-versus-rest schemes. - \'newton-cg\', \'lbfgs\' and \'sag\' only handle L2 penalty.  Note that \'sag\' fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.  .. versionadded:: 0.17 Stochastic Average Gradient descent solver.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    tol = hyperparams.Bounded[float](
        default=0.0001,
        lower=0,
        upper=None,
        description='Tolerance for stopping criteria.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    C = hyperparams.Hyperparameter[float](
        default=1.0,
        description='Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    multi_class = hyperparams.Enumeration[str](
        values=['ovr', 'multinomial'],
        default='ovr',
        description='Multiclass option can be either \'ovr\' or \'multinomial\'. If the option chosen is \'ovr\', then a binary problem is fit for each label. Else the loss minimised is the multinomial loss fit across the entire probability distribution. Works only for the \'newton-cg\', \'sag\' and \'lbfgs\' solver.  .. versionadded:: 0.18 Stochastic Average Gradient descent solver for \'multinomial\' case.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    warm_start = hyperparams.UniformBool(
        default=False,
        description='When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver.  .. versionadded:: 0.17 *warm_start* to support *lbfgs*, *newton-cg*, *sag* solvers.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_jobs = hyperparams.Union(
        configuration=OrderedDict({
            'limit': hyperparams.Bounded[int](
                default=1,
                lower=1,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'all_cores': hyperparams.Constant(
                default=-1,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='limit',
        description='Number of CPU cores used during the cross-validation loop. If given a value of -1, all cores are used.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter']
    )
    
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



class SKLogisticRegression(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                          ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn LogisticRegression
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
    
    """
    
    __author__ = "JPL MARVIN"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LOGISTIC_REGRESSION, ],
         "name": "sklearn.linear_model.logistic.LogisticRegression",
         "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
         "python_path": "d3m.primitives.classification.logistic_regression.SKlearn",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html']},
         "version": "v2019.2.27",
         "id": "b9c81b40-8ed1-3b23-80cf-0d6fe6863962",
         'installation': [
                        # TODO : Will update based on https://gitlab.com/datadrivendiscovery/d3m/issues/137
                        #{
                        #    "type": "PIP",
                        #    "package_uri": "git+https://gitlab.com/datadrivendiscovery/common-primitives.git@26419dde2f660f901066c896a972ae4c438ee236#egg=common_primitives"
                        #},
                        {'type': metadata_base.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@{git_commit}#egg=sklearn_wrap'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                            ),
                           }]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = LogisticRegression(
              penalty=self.hyperparams['penalty'],
              dual=self.hyperparams['dual'],
              fit_intercept=self.hyperparams['fit_intercept'],
              intercept_scaling=self.hyperparams['intercept_scaling'],
              class_weight=self.hyperparams['class_weight'],
              max_iter=self.hyperparams['max_iter'],
              solver=self.hyperparams['solver'],
              tol=self.hyperparams['tol'],
              C=self.hyperparams['C'],
              multi_class=self.hyperparams['multi_class'],
              warm_start=self.hyperparams['warm_start'],
              n_jobs=self.hyperparams['n_jobs'],
              random_state=self.random_seed,
              verbose=_verbose
        )
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

        self._clf.fit(self._training_inputs, sk_training_output)
        self._fitted = True

        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        sk_output = self._clf.predict(sk_inputs)
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
                coef_=None,
                intercept_=None,
                n_iter_=None,
                classes_=None,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            coef_=getattr(self._clf, 'coef_', None),
            intercept_=getattr(self._clf, 'intercept_', None),
            n_iter_=getattr(self._clf, 'n_iter_', None),
            classes_=getattr(self._clf, 'classes_', None),
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_columns_metadata_=self._target_columns_metadata,
            target_column_indices_=self._target_column_indices
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.coef_ = params['coef_']
        self._clf.intercept_ = params['intercept_']
        self._clf.n_iter_ = params['n_iter_']
        self._clf.classes_ = params['classes_']
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
        if params['classes_'] is not None:
            self._fitted = True
    def log_likelihoods(self, *,
                    outputs: Outputs,
                    inputs: Inputs,
                    timeout: float = None,
                    iterations: int = None) -> CallResult[Sequence[float]]:
        inputs = inputs.values  # Get ndarray
        outputs = outputs.values
        return CallResult(self._clf.predict_log_proba(inputs)[:, outputs])
    


    
    
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


SKLogisticRegression.__doc__ = LogisticRegression.__doc__