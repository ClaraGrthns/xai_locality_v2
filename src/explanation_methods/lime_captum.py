from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import LimeBase
import torch

from torch import Tensor

import os.path as osp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from captum._utils.models.linear_model import SkLearnLinearModel
from captum.attr._core.lime import LimeBase
#!/usr/bin/env python3
import inspect
import math
import warnings
from typing import Any, Callable, cast, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
)
from captum._utils.progress import progress
from captum._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.log import log_usage
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
#!/usr/bin/env python3

# pyre-strict
import inspect
import math
import typing
import warnings
from collections.abc import Iterator
from typing import Any, Callable, cast, List, Literal, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _flatten_tensor_or_tuple,
    _format_output,
    _format_tensor_into_tuples,
    _get_max_feature_index,
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.models.model import Model
from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import (
    _construct_default_feature_mask,
    _format_input_baseline,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader, TensorDataset


from captum.attr._core.lime import get_exp_kernel_similarity_function, default_perturb_func, default_from_interp_rep_transform, construct_feature_mask 

class LimeCaptumHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs_explainer):
        model = kwargs_explainer.get("model")
        def to_interp_transform(curr_sample, 
                                original_inp, 
                                **kwargs):
            return curr_sample
        def perturb_func(original_input: Tensor,
                         **kwargs)->Tensor:
            return original_input + torch.randn_like(original_input)        
        
        def similarity_kernel(original_input: Tensor,
                              perturbed_input: Tensor,
                              perturbed_interpretable_input: Tensor, 
                              **kwargs)->Tensor:
            # kernel_width will be provided to attribute as a kwarg
            kernel_width = kwargs_explainer.get("kernel_width", np.sqrt(original_input.shape[1])*0.75)
            l2_dist = torch.norm(original_input - perturbed_input)
            return torch.exp(- (l2_dist**2) / (kernel_width**2))
        
        self.explainer = LimeWithBias(model, 
                                    SkLearnLinearModel("linear_model.Ridge"),
                                    similarity_func=similarity_kernel,
                                    perturb_func=perturb_func,
                                    perturb_interpretable_space=False,
                                    from_interp_rep_transform=None,
                                    to_interp_rep_transform=to_interp_transform)

    def explain_instance(self, **kwargs):
        coefs, bias = self.explainer.attribute(kwargs["input"], target=kwargs["target"])
        return coefs, bias
    
    def compute_explanations(self, results_path, predict_fn, tst_data, tst_set=True):
        tst_feat_for_expl_loader = DataLoader(tst_data, batch_size=1, shuffle=False)
        device = torch.device("cpu")
        feature_attribution_folder = osp.join(results_path,
                                    "feature_attribution")
        
        coefs_feature_attribution_file_path = osp.join(feature_attribution_folder, f"coefs_feature_attribution_kernel_width-{self.args.kernel_width}.h5")

        coefs_feature_attributions = None
        print("Looking for LIME explanations (coefficients and bias) in: ", feature_attribution_folder)
        # Check if both files exist and force is not set
        should_load = osp.exists(coefs_feature_attribution_file_path)
        if should_load:
            print(f"Using precomputed LIME explanations from: {feature_attribution_folder}")
            try:
                with h5py.File(coefs_feature_attribution_file_path, "r") as f:
                    coefs_feature_attributions = f["coefs_feature_attribution"][:]
                coefs_feature_attributions = torch.tensor(coefs_feature_attributions).float().to(device)
                print("Successfully loaded precomputed explanations.")
            except Exception as e:
                print(f"Error loading precomputed explanations: {e}. Recomputing...")
                coefs_feature_attributions = None
                should_load = False # Force recomputation

        if not should_load:
            print("Precomputed LIME explanations not found or loading failed/forced. Computing explanations for the test set...")
            if not osp.exists(feature_attribution_folder):
                os.makedirs(feature_attribution_folder)
            coefs = self.compute_feature_attributions(self.explainer, predict_fn, tst_feat_for_expl_loader)
            coefs_feature_attributions = coefs.float().to(device)
            print(f"Saving computed coefficients to: {coefs_feature_attribution_file_path}")
            with h5py.File(coefs_feature_attribution_file_path, "w") as f:
                f.create_dataset("coefs_feature_attribution", data=coefs_feature_attributions.cpu().numpy())
            print("Finished computing and saving explanations.")
        if coefs_feature_attributions is None:
             raise RuntimeError("Failed to load or compute LIME explanations.")
        return coefs_feature_attributions
    

    def compute_feature_attributions(self, explainer, predict_fn, data_loader_tst, transform = None):
        coefs_feature_attribution = []
        for i, batch in enumerate(data_loader_tst):
            Xs = batch#[0]
            preds = predict_fn(Xs)
            if preds.ndim == 1 or preds.shape[1] == 1:
                coefs, bias = explainer.attribute(Xs, return_input_shape=True)
                coefs = coefs.float()
            else:
                top_labels = torch.argmax(predict_fn(Xs), dim=1).tolist()
                coefs, bias = explainer.attribute(Xs, target=top_labels, return_input_shape=True)
                coefs = coefs.float()
            coefs_feature_attribution.append(coefs)
            print("computed the first stack of feature_attribution maps")
        return torch.cat(coefs_feature_attribution, dim=0)

    def get_experiment_setting(self, n_nearest_neighbors):
        df_setting = "dataset_test"
        df_setting += "_val" if self.args.include_val else ""
        df_setting += "_trn" if self.args.include_trn else ""
        if self.args.kernel_width == "default":
            setting = f"{self.args.method}_{df_setting}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_random_seed-{self.args.random_seed}_difference_vs_kNN"
        else:
            setting = f"{self.args.method}_{df_setting}_kernel_width-{self.args.kernel_width}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_random_seed-{self.args.random_seed}_difference_vs_kNN"
        setting = f"kNN-1-{np.round(n_nearest_neighbors, 2)}_"+setting
        if self.args.regression:
            setting = setting + "_regression" 
        return setting
    



class LimeWithBias(LimeBase):
    """
    A wrapper class for captum.attr.LimeBase that modifies the attribute
    method to return both the interpretable model representation (e.g., coefficients)
    and the bias (intercept) term of the fitted surrogate model.

    Assumes the provided `interpretable_model` stores the fitted model
    in a way that the intercept can be accessed (e.g., via a `.model.intercept_`
    attribute, common when using SkLearnLinearModel wrapper).
    """

    def __init__(self, forward_func, interpretable_model, **kwargs):
        """
        Initializes LimeWithBias.

        Args:
            forward_func (callable): The forward function of the model or
                        any modification of it.
            interpretable_model (callable): A function or model instance that
                        takes inputs (interpretable input, original output,
                        weights), trains an interpretable model, and returns
                        a representation of the interpretable model. Common examples
                        include wrappers around sklearn.linear_model (e.g.,
                        captum._utils.models.linear_model.SkLearnLinearModel).
            **kwargs: Additional arguments are passed to the LimeBase constructor.
                      Refer to LimeBase documentation for details (similarity_func,
                      perturb_func, etc.).
        """
        super().__init__(forward_func, interpretable_model, **kwargs)
    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above.
        It trains an interpretable model and returns a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        A batch of inputs can be provided as inputs only if forward_func
        returns a single value per batch (e.g. loss).
        The interpretable feature representation should still have shape
        1 x num_interp_features, corresponding to the interpretable
        representation for the full batch, and perturbations_per_eval
        must be set to 1.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. For all other types,
                        the given argument is used for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_samples (int, optional): The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False
            **kwargs (Any, optional): Any additional arguments necessary for
                        sampling and transformation functions (provided to
                        constructor).
                        Default: None

        Returns:
            **interpretable model representation**:
            - **interpretable model representation** (*Any*):
                    A representation of the interpretable model trained. The return
                    type matches the return type of train_interpretable_model_func.
                    For example, this could contain coefficients of a
                    linear surrogate model.

        Examples::

            >>> # SimpleClassifier takes a single input tensor of
            >>> # float features with size N x 5,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>>
            >>> # We will train an interpretable model with the same
            >>> # features by simply sampling with added Gaussian noise
            >>> # to the inputs and training a model to predict the
            >>> # score of the target class.
            >>>
            >>> # For interpretable model training, we will use sklearn
            >>> # linear model in this example. We have provided wrappers
            >>> # around sklearn linear models to fit the Model interface.
            >>> # Any arguments provided to the sklearn constructor can also
            >>> # be provided to the wrapper, e.g.:
            >>> # SkLearnLinearModel("linear_model.Ridge", alpha=2.0)
            >>> from captum._utils.models.linear_model import SkLearnLinearModel
            >>>
            >>>
            >>> # Define similarity kernel (exponential kernel based on L2 norm)
            >>> def similarity_kernel(
            >>>     original_input: Tensor,
            >>>     perturbed_input: Tensor,
            >>>     perturbed_interpretable_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         # kernel_width will be provided to attribute as a kwarg
            >>>         kernel_width = kwargs["kernel_width"]
            >>>         l2_dist = torch.norm(original_input - perturbed_input)
            >>>         return torch.exp(- (l2_dist**2) / (kernel_width**2))
            >>>
            >>>
            >>> # Define sampling function
            >>> # This function samples in original input space
            >>> def perturb_func(
            >>>     original_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         return original_input + torch.randn_like(original_input)
            >>>
            >>> # For this example, we are setting the interpretable input to
            >>> # match the model input, so the to_interp_rep_transform
            >>> # function simply returns the input. In most cases, the interpretable
            >>> # input will be different and may have a smaller feature set, so
            >>> # an appropriate transformation function should be provided.
            >>>
            >>> def to_interp_transform(curr_sample, original_inp,
            >>>                                      **kwargs):
            >>>     return curr_sample
            >>>
            >>> # Generating random input with size 1 x 5
            >>> input = torch.randn(1, 5)
            >>> # Defining LimeBase interpreter
            >>> lime_attr = LimeBase(net,
                                     SkLearnLinearModel("linear_model.Ridge"),
                                     similarity_func=similarity_kernel,
                                     perturb_func=perturb_func,
                                     perturb_interpretable_space=False,
                                     from_interp_rep_transform=None,
                                     to_interp_rep_transform=to_interp_transform)
            >>> # Computes interpretable model, returning coefficients of linear
            >>> # model.
            >>> attr_coefs = lime_attr.attribute(input, target=1, kernel_width=1.1)
        """
        with torch.no_grad():
            inp_tensor = (
                cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
            )
            device = inp_tensor.device

            interpretable_inps = []
            similarities = []
            outputs = []

            curr_model_inputs = []
            expanded_additional_args = None
            expanded_target = None
            perturb_generator = None
            if inspect.isgeneratorfunction(self.perturb_func):
                perturb_generator = self.perturb_func(inputs, **kwargs)

            if show_progress:
                attr_progress = progress(
                    total=math.ceil(n_samples / perturbations_per_eval),
                    desc=f"{self.get_name()} attribution",
                )
                attr_progress.update(0)

            batch_count = 0
            for _ in range(n_samples):
                if perturb_generator:
                    try:
                        curr_sample = next(perturb_generator)
                    except StopIteration:
                        warnings.warn(
                            "Generator completed prior to given n_samples iterations!"
                        )
                        break
                else:
                    curr_sample = self.perturb_func(inputs, **kwargs)
                batch_count += 1
                if self.perturb_interpretable_space:
                    interpretable_inps.append(curr_sample)
                    curr_model_inputs.append(
                        self.from_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                else:
                    curr_model_inputs.append(curr_sample)
                    interpretable_inps.append(
                        self.to_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                curr_sim = self.similarity_func(
                    inputs, curr_model_inputs[-1], interpretable_inps[-1], **kwargs
                )
                similarities.append(
                    curr_sim.flatten()
                    if isinstance(curr_sim, Tensor)
                    else torch.tensor([curr_sim], device=device)
                )

                if len(curr_model_inputs) == perturbations_per_eval:
                    if expanded_additional_args is None:
                        expanded_additional_args = _expand_additional_forward_args(
                            additional_forward_args, len(curr_model_inputs)
                        )
                    if expanded_target is None:
                        expanded_target = _expand_target(target, len(curr_model_inputs))

                    model_out = self._evaluate_batch(
                        curr_model_inputs,
                        expanded_target,
                        expanded_additional_args,
                        device,
                    )

                    if show_progress:
                        attr_progress.update()

                    outputs.append(model_out)

                    curr_model_inputs = []

            if len(curr_model_inputs) > 0:
                expanded_additional_args = _expand_additional_forward_args(
                    additional_forward_args, len(curr_model_inputs)
                )
                expanded_target = _expand_target(target, len(curr_model_inputs))
                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )
                if show_progress:
                    attr_progress.update()
                outputs.append(model_out)

            if show_progress:
                attr_progress.close()

            combined_interp_inps = torch.cat(interpretable_inps).float()
            combined_outputs = (
                torch.cat(outputs)
                if len(outputs[0].shape) > 0
                else torch.stack(outputs)
            ).float()
            combined_sim = (
                torch.cat(similarities)
                if len(similarities[0].shape) > 0
                else torch.stack(similarities)
            ).float()
            dataset = TensorDataset(
                combined_interp_inps, combined_outputs, combined_sim
            )
            self.interpretable_model.fit(DataLoader(dataset, batch_size=batch_count))
            return self.interpretable_model.representation(), self.interpretable_model.bias()




class Lime(LimeWithBias):
    r"""
    Lime is an interpretability method that trains an interpretable surrogate model
    by sampling points around a specified input example and using model evaluations
    at these points to train a simpler interpretable 'surrogate' model, such as a
    linear model.

    Lime provides a more specific implementation than LimeBase in order to expose
    a consistent API with other perturbation-based algorithms. For more general
    use of the LIME framework, consider using the LimeBase class directly and
    defining custom sampling and transformation to / from interpretable
    representation functions.

    Lime assumes that the interpretable representation is a binary vector,
    corresponding to some elements in the input being set to their baseline value
    if the corresponding binary interpretable feature value is 0 or being set
    to the original input value if the corresponding binary interpretable
    feature value is 1. Input values can be grouped to correspond to the same
    binary interpretable feature using a feature mask provided when calling
    attribute, similar to other perturbation-based attribution methods.

    One example of this setting is when applying Lime to an image classifier.
    Pixels in an image can be grouped into super-pixels or segments, which
    correspond to interpretable features, provided as a feature_mask when
    calling attribute. Sampled binary vectors convey whether a super-pixel
    is on (retains the original input values) or off (set to the corresponding
    baseline value, e.g. black image). An interpretable linear model is trained
    with input being the binary vectors and outputs as the corresponding scores
    of the image classifier with the appropriate super-pixels masked based on the
    binary vector. Coefficients of the trained surrogate
    linear model convey the importance of each super-pixel.

    More details regarding LIME can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        interpretable_model: Optional[Model] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        similarity_func: Optional[Callable] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        perturb_func: Optional[Callable] = None,
    ) -> None:
        r"""

        Args:


            forward_func (Callable): The forward function of the model or any
                    modification of it
            interpretable_model (Model, optional): Model object to train
                    interpretable model.

                    This argument is optional and defaults to SkLearnLasso(alpha=0.01),
                    which is a wrapper around the Lasso linear model in SkLearn.
                    This requires having sklearn version >= 0.23 available.

                    Other predefined interpretable linear models are provided in
                    captum._utils.models.linear_model.

                    Alternatively, a custom model object must provide a `fit` method to
                    train the model, given a dataloader, with batches containing
                    three tensors:

                    - interpretable_inputs: Tensor
                      [2D num_samples x num_interp_features],
                    - expected_outputs: Tensor [1D num_samples],
                    - weights: Tensor [1D num_samples]

                    The model object must also provide a `representation` method to
                    access the appropriate coefficients or representation of the
                    interpretable model after fitting.

                    Note that calling fit multiple times should retrain the
                    interpretable model, each attribution call reuses
                    the same given interpretable model object.
            similarity_func (Callable, optional): Function which takes a single sample
                    along with its corresponding interpretable representation
                    and returns the weight of the interpretable sample for
                    training the interpretable model.
                    This is often referred to as a similarity kernel.

                    This argument is optional and defaults to a function which
                    applies an exponential kernel to the cosine distance between
                    the original input and perturbed input, with a kernel width
                    of 1.0.

                    A similarity function applying an exponential
                    kernel to cosine / euclidean distances can be constructed
                    using the provided get_exp_kernel_similarity_function in
                    captum.attr._core.lime.

                    Alternately, a custom callable can also be provided.
                    The expected signature of this callable is:

                    >>> def similarity_func(
                    >>>    original_input: Tensor or tuple[Tensor, ...],
                    >>>    perturbed_input: Tensor or tuple[Tensor, ...],
                    >>>    perturbed_interpretable_input:
                    >>>        Tensor [2D 1 x num_interp_features],
                    >>>    **kwargs: Any
                    >>> ) -> float or Tensor containing float scalar

                    perturbed_input and original_input will be the same type and
                    contain tensors of the same shape, with original_input
                    being the same as the input provided when calling attribute.

                    kwargs includes baselines, feature_mask, num_interp_features
                    (integer, determined from feature mask).
            perturb_func (Callable, optional): Function which returns a single
                    sampled input, which is a binary vector of length
                    num_interp_features, or a generator of such tensors.

                    This function is optional, the default function returns
                    a binary vector where each element is selected
                    independently and uniformly at random. Custom
                    logic for selecting sampled binary vectors can
                    be implemented by providing a function with the
                    following expected signature:

                    >>> perturb_func(
                    >>>    original_input: Tensor or tuple[Tensor, ...],
                    >>>    **kwargs: Any
                    >>> ) -> Tensor [Binary 2D Tensor 1 x num_interp_features]
                    >>>  or generator yielding such tensors

                    kwargs includes baselines, feature_mask, num_interp_features
                    (integer, determined from feature mask).

        """
        if interpretable_model is None:
            interpretable_model = SkLearnLasso(alpha=0.01)

        if similarity_func is None:
            similarity_func = get_exp_kernel_similarity_function()

        if perturb_func is None:
            perturb_func = default_perturb_func

        LimeBase.__init__(
            self,
            forward_func,
            interpretable_model,
            similarity_func,
            perturb_func,
            True,
            default_from_interp_rep_transform,
            None,
        )

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above,
        training an interpretable model and returning a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        A batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, if forward_fn
        returns a scalar per example, attributions will be computed for each
        example independently, with a separate interpretable model trained for each
        example. Note that provided similarity and perturbation functions will be
        provided each example separately (first dimension = 1) in this case.
        If forward_fn returns a scalar per batch (e.g. loss), attributions will
        still be computed using a single interpretable model for the full batch.
        In this case, similarity and perturbation functions will be provided the
        same original input containing the full batch.

        The number of interpretable features is determined from the provided
        feature mask, or if none is provided, from the default feature mask,
        which considers each scalar input as a separate feature. It is
        generally recommended to provide a feature mask which groups features
        into a small number of interpretable features / components (e.g.
        superpixels in images).

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference value which replaces each
                        feature when the corresponding interpretable feature
                        is set to 0.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.
                        Default: None
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            feature_mask (Tensor or tuple[Tensor, ...], optional):
                        feature_mask defines a mask for the input, grouping
                        features which correspond to the same
                        interpretable feature. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Values across
                        all tensors should be integers in the range 0 to
                        num_interp_features - 1, and indices corresponding to the
                        same feature should have the same value.
                        Note that features are grouped across tensors
                        (unlike feature ablation and occlusion), so
                        if the same index is used in different tensors, those
                        features are still grouped and added simultaneously.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature.
                        Default: None
            n_samples (int, optional): The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            return_input_shape (bool, optional): Determines whether the returned
                        tensor(s) only contain the coefficients for each interp-
                        retable feature from the trained surrogate model, or
                        whether the returned attributions match the input shape.
                        When return_input_shape is True, the return type of attribute
                        matches the input shape, with each element containing the
                        coefficient of the corresponding interpretale feature.
                        All elements with the same value in the feature mask
                        will contain the same coefficient in the returned
                        attributions. If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
                        num_interp_features.
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The attributions with respect to each input feature.
                        If return_input_shape = True, attributions will be
                        the same size as the provided inputs, with each value
                        providing the coefficient of the corresponding
                        interpretale feature.
                        If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
                        num_interp_features.
        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()

            >>> # Generating random input with size 1 x 4 x 4
            >>> input = torch.randn(1, 4, 4)

            >>> # Defining Lime interpreter
            >>> lime = Lime(net)
            >>> # Computes attribution, with each of the 4 x 4 = 16
            >>> # features as a separate interpretable feature
            >>> attr = lime.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we can group each 2x2 square of the inputs
            >>> # as one 'interpretable' feature and perturb them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are set to their
            >>> # baseline value, when the corresponding binary interpretable
            >>> # feature is set to 0.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # Computes interpretable model and returning attributions
            >>> # matching input shape.
            >>> attr = lime.attribute(input, target=1, feature_mask=feature_mask)
        """
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        return super().attribute_future()
 # --- Step 5: Copy and Modify _attribute_kwargs ---
    def _attribute_kwargs(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
        **kwargs: object,
    # --- CHANGE: Update return type hint ---
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Any]:

        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        bsz = formatted_inputs[0].shape[0]

        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. ",
                stacklevel=1,
            )

        # --- CHANGE: Prepare lists to store both coefs and biases for batches ---
        output_rep_list: List = []
        output_bias_list: List = []

        if bsz > 1:
            test_output = _run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )
            is_multi_output_per_example = (
                isinstance(test_output, Tensor) and torch.numel(test_output) > 1
            )

            if is_multi_output_per_example:
                if torch.numel(test_output) == bsz:
                    warnings.warn(
                        "You are providing multiple inputs for Lime. This trains a "
                        "separate interpretable model for each example, which can be "
                        "time consuming. It is recommended to compute attributions "
                        "for one example at a time.",
                        stacklevel=1,
                    )
                    # --- Batch Processing Loop ---
                    for (
                        curr_inps,
                        curr_target,
                        curr_additional_args,
                        curr_baselines,
                        curr_feature_mask,
                    ) in _batch_example_iterator(
                        bsz,
                        formatted_inputs,
                        target,
                        additional_forward_args,
                        baselines,
                        feature_mask,
                    ):
                        # --- CHANGE: Capture both return values ---
                        # Note: Using super().attribute directly now, assuming it's overridden in LimeWithBias
                        curr_coefs, curr_bias = super().attribute(
                            inputs=curr_inps if is_inputs_tuple else curr_inps[0],
                            target=curr_target,
                            additional_forward_args=curr_additional_args,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            # Pass kwargs needed by LimeBase/LimeWithBias attribute
                            baselines=(
                                curr_baselines if is_inputs_tuple else curr_baselines[0]
                            ),
                            feature_mask=(
                                curr_feature_mask
                                if is_inputs_tuple
                                else curr_feature_mask[0]
                            ),
                            num_interp_features=num_interp_features,
                            show_progress=False, # Progress handled outside loop if needed
                            **kwargs, # Pass any other relevant kwargs
                        )
                        # --- Store bias ---
                        output_bias_list.append(curr_bias)

                        # --- Process coefs based on return_input_shape ---
                        if return_input_shape:
                            processed_coefs = self._convert_output_shape(
                                curr_inps, # Pass single example input
                                curr_feature_mask, # Pass single example mask
                                curr_coefs,
                                num_interp_features,
                                is_inputs_tuple,
                            )
                            output_rep_list.append(processed_coefs)
                        else:
                             # Ensure coefs is a tensor before reshape
                             if not isinstance(curr_coefs, Tensor):
                                 curr_coefs = torch.tensor(curr_coefs) # Or handle error
                             output_rep_list.append(curr_coefs.reshape(1, -1))
                    # --- End Batch Loop ---

                    # --- CHANGE: Combine results ---
                    combined_reps = _reduce_list(output_rep_list)
                    # Combine biases (stack if possible, otherwise return list)
                    try:
                        # Attempt to stack if they are tensors of compatible shape
                        if all(isinstance(b, Tensor) for b in output_bias_list):
                            combined_biases = torch.stack(output_bias_list)
                        # Convert scalars/numpy arrays to tensor
                        elif all(isinstance(b, (int, float)) or type(b).__module__ == 'numpy' for b in output_bias_list):
                             device = combined_reps[0].device if _is_tuple(combined_reps) else combined_reps.device
                             combined_biases = torch.tensor(output_bias_list, device=device)
                        else:
                             combined_biases = output_bias_list # Return list if mixed types
                    except Exception as e:
                        warnings.warn(f"Could not stack biases: {e}. Returning a list of biases.")
                        combined_biases = output_bias_list

                    return combined_reps, combined_biases # Return tuple
                else:
                     raise AssertionError(
                        "Invalid number of outputs, forward function should return a"
                        "scalar per example or a scalar per input batch."
                    )
            # --- End multi-output-per-example case ---
            else: # Single output for the whole batch
                if perturbations_per_eval != 1:
                    raise AssertionError(
                        "Perturbations per eval must be 1 when forward function "
                        "returns single value per batch!"
                    )
                # Fall through to single attribution call below

        # --- Single Attribution Call (bsz=1 or single output for batch) ---
        # --- CHANGE: Capture both return values ---
        coefs, bias = super().attribute( # Assumes LimeWithBias.attribute
            inputs=inputs, # Pass original potentially batched inputs
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
             # Pass kwargs needed by LimeBase/LimeWithBias attribute
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            show_progress=show_progress,
            **kwargs, # Pass any other relevant kwargs
        )

        # --- Process coefs based on return_input_shape ---
        if return_input_shape:
            shaped_reps = self._convert_output_shape(
                formatted_inputs, # Pass original formatted inputs
                feature_mask,     # Pass original feature mask
                coefs,
                num_interp_features,
                is_inputs_tuple,
            )
            # --- CHANGE: Return tuple ---
            return shaped_reps, bias
        else:
            # --- CHANGE: Return tuple ---
            return coefs, bias
        

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[True],
    ) -> Tuple[Tensor, ...]: ...

    @typing.overload
    def _convert_output_shape(  # type: ignore
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[False],
    ) -> Tensor: ...

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...]]: ...

    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        coefs = coefs.flatten()
        attr = [
            torch.zeros_like(single_inp, dtype=torch.float)
            for single_inp in formatted_inp
        ]
        for tensor_ind in range(len(formatted_inp)):
            for single_feature in range(num_interp_features):
                attr[tensor_ind] += (
                    coefs[single_feature].item()
                    * (feature_mask[tensor_ind] == single_feature).float()
                )
        return _format_output(is_inputs_tuple, tuple(attr))
