import copy
from typing import Any, Mapping
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from .grid import GridND,UniformDistribution
from .graph import IntegralTracer
from .integral_group import IntegralGroup
from .parametrizations.base_parametrization import IWeights
from .parametrizations import GridSampleWeights1D
from .parametrizations import GridSampleWeights2D
from .parametrizations import InterpolationWeights1D
from .parametrizations import InterpolationWeights2D
from .permutation import NOptPermutation
from .quadrature import TrapezoidalQuadrature
from .grid import TrainableGrid1D
from .utils import (
    remove_parametrizations,
    reapply_parametrizations,
    reset_batchnorm,
    get_parent_name,
    fuse_batchnorm,
    get_parent_module,
)
import numpy as np
from numpy import linalg

class ParametrizedModel(nn.Module):
    def __init__(self, model):
        super(ParametrizedModel, self).__init__()
        self.model = model
        self.threshold = 0.92
    def reset_rate(self):
        for g in self.groups:
            g.rate = 1
    def check_rate(self):
        for g in self.groups:
            if g.typeo == "normal":
                if int(g.ori_size*g.rate)%g.base!=0:
                    g.rate = int(int(g.ori_size*g.rate)/g.base + 0.5)*g.base/g.ori_size
    def set_size(self):
        self.check_rate()
        for g in self.groups:
            if g.typeo != "normal":
                rate = g.partner.rate
            else:
                rate = g.rate
            if g.size < 4:
                size = g.ori_size
            else:
                size = g.ori_size*rate
                print("now",g.base,size,rate)
            size = int(size)
            if g.typeo != "max":
                if size%g.base==0:
                    g.resize(int(size))
                else:
                    g.resize(int(int(size/g.base+1)*(g.base)))
    def set_distributions(self):
        for g in self.groups:
            if g.typeo != "normal":
                rate = g.partner.rate
            else:
                rate = g.rate
            g.reset_distribution(UniformDistribution(int(g.ori_size*rate),int(g.ori_size),g.base))
    def set_rate(self):
        self.eval()
        for g in self.groups:
            g.reset_distribution(UniformDistribution(int(g.ori_size),int(g.ori_size),g.base))
            size = g.ori_size
            if g.typeo != "max":
                g.resize(int(size))
        self.eval()
        self.forward_groups()
        upm = self.get_unparametrized_model()
        for g in self.groups:
            if g.typeo == 'normal':
                print("---------------")
                pined = False
                rate = 1
                for param in g.params:
                    if 'fc' in param['name'] or 'conv00' in param['name']:
                        continue
                    nl = param['name'].split(".")
                    tg = upm
                    for i in nl:
                        tg = getattr(tg, i)
                    if len(tg.shape) != 4:
                        continue
                    mats = []
                    if ('core' in param['name']) and ('core_proj' not in param['name']):
                        if tg.shape[0] < tg.shape[1]:
                            for i in range(tg.shape[1]//tg.shape[0]):
                                mats.append(tg[:,i*tg.shape[0]:(i+1)*tg.shape[0],:,:])
                        else:
                            for i in range(tg.shape[0]//tg.shape[1]):
                                mats.append(tg[i*tg.shape[1]:(i+1)*tg.shape[1],:,:,:])
                    else:
                        mats.append(tg)
                    nowrate = 1
                    for i in range(tg.shape[-2]):
                        for j in range(tg.shape[-1]):
                            for mat in mats:
                                _,now,_ = torch.linalg.svd(mat[:,:,i,j])
                                now = sorted(now,reverse=True)
                                total = sum([k*k for k in now])
                                tmp = 0
                                idx = 0
                                for k in now:
                                    idx += 1
                                    tmp += k*k
                                    if tmp/total > self.threshold:#lower maybe
                                        nowrate = min(nowrate,idx/len(now))
                                        # nowrate = max([u*u for u in now])/total
                                        break
                    if ('core' in param['name']) and ('core_proj' not in param['name']):
                        rate = min(nowrate,rate)
                        pined = True
                    elif not pined:
                        rate = min(rate, nowrate)
                    print(param['name'],rate)
                g.rate = rate
        for g in self.groups:
            if g.typeo != "max":
                g.resize(int(g.ori_size))
        self.check_rate()
    def forward(self, *args, **kwargs):
        """ """
        #self.forward_groups()
        return self.model(*args, **kwargs)
    def forward_groups(self):
        pass
    def get_unparametrized_model(self):
        """Samples weights, removes parameterizations and returns discrete model."""
        self.forward_groups()
        parametrized_modules = remove_parametrizations(self.model)
        unparametrized_model = copy.deepcopy(self.model)
        reapply_parametrizations(self.model, parametrized_modules, True)

        return unparametrized_model

    def __getattr__(self, item):
        if item in dir(self):
            out = super().__getattr__(item)
        else:
            out = getattr(self.model, item)

        return out

    def __getstate__(self):
        """
        Return the state of the module, removing the non-picklable parametrizations.
        """
        parametrized_modules = remove_parametrizations(self.model)
        state = self.state_dict()
        state["parametrized_modules"] = parametrized_modules

        return state

    def __setstate__(self, state):
        """Initialize the module from its state."""
        parametrized_modules = state.pop("parametrized_modules")
        super().__setstate__(state)
        reapply_parametrizations(self.model, parametrized_modules, True)


class PrunableModel(ParametrizedModel):
    def __init__(self, model, groups):
        super(PrunableModel, self).__init__(model)
        groups.sort(key=lambda g: g.count_parameters())
        self.groups = nn.ModuleList(groups)

    def forward_groups(self):
        for g in self.groups:
            g.vis = False
        for group in self.groups:
            if group.vis:
                continue
            if group.typeo == "normal":
                group()
            else:
                tmp = group.partner
                while ((tmp is not None) and (not tmp.vis)):
                    tmp()
                    tmp = tmp.partner
                group()
        for group in self.groups:
            group.vis = False
                    



class IntegralModel(PrunableModel):
    """
    Contains original model with parametrized layers and RelatedGroups list.

    Parameters
    ----------
    model: torch.nn.Module.
        Model with parametrized layers.
    groups: List[RelatedGroup].
        List related groups.
    """

    def __init__(self, model, groups):
        super(IntegralModel, self).__init__(model, groups)
        self.original_size = None
        self.original_size = self.calculate_compression()

    def clear(self):
        """Clears cached tensors in all integral groups."""
        for group in self.groups:
            group.clear()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        out = super().load_state_dict(state_dict, strict)
        self.clear()

        return out

    def calculate_compression(self):
        """
        Returns 1 - ratio of the size of the current
        model to the original size of the model.
        """
        self.forward_groups()

        for group in self.groups:
            group.clear()

        parametrized = remove_parametrizations(self.model)
        out = sum(p.numel() for p in self.model.parameters())
        reapply_parametrizations(self.model, parametrized, True)

        if self.original_size is not None:
            out = 1.0 - out / self.original_size

        return out

    def resize(self, sizes):
        """
        Resizes grids in each group.

        Parameters
        ----------
        sizes: List[int].
            List of new sizes.
        """
        for group, size in zip(self.groups, sizes):
            group.resize(size)

    def reset_grids(self, grids):
        for group, grid in zip(self.groups, grids):
            group.reset_grid(grid)

    def reset_distributions(self, distributions):
        """
        Sets new distributions in each RelatedGroup.grid.

        Parameters
        ----------
        distributions: List[torch_integral.grid.Distribution].
            List of new distributions.
        """
        for group, dist in zip(self.groups, distributions):
            group.reset_distribution(dist)

    def grids(self):
        """Returns list of grids of each integral group."""
        return [group.grid for group in self.groups]

    def grid_tuning(self, train_bn=False, train_bias=False, use_all_grids=False):
        """Turns on grid tuning mode for fast post-training pruning.
        Sets requires_grad = False for all parameters except TrainableGrid's parameters,
        biases and BatchNorm parameters (if corresponding flag is True).

        Parameters
        ----------
        train_bn: bool.
            Set True to train BatchNorm parameters.
        train_bias: bool.
            Set True to train biases.
        use_all_grids: bool.
            Set True to use all grids in each group.
        """
        if use_all_grids:
            for group in self.groups:
                if group.subgroups is None:
                    group.reset_grid(TrainableGrid1D(group.grid_size()))

        for name, param in self.named_parameters():
            parent = get_parent_module(self, name)

            if isinstance(parent, TrainableGrid1D):
                param.requires_grad = True
            else:
                param.requires_grad = False

        if train_bn:
            reset_batchnorm(self)

        if train_bias:
            for group in self.groups:
                for p in group.params:
                    if "bias" in p["name"]:
                        parent = get_parent_module(self.model, p["name"])

                        if parametrize.is_parametrized(parent, "bias"):
                            parametrize.remove_parametrizations(parent, "bias", True)

                        getattr(parent, "bias").requires_grad = True


class IntegralWrapper:
    """
    Wrapper class which allows batch norm fusion,
    permutation of tensor parameters to obtain continuous structure in the tensor
    and convertation of discrete model to integral.

    Parameters
    ----------
    init_from_discrete: bool.
        If set True, then parametrization will be optimized with
        gradient descent to approximate discrete model's weights.
    fuse_bn: bool.
        If True, then convolutions and batchnorms will be fused.
    optimize_iters: int.
        Number of optimization iterations for discerete weight tensor approximation.
    start_lr: float.
        Learning rate when optimizing parametrizations.
    permutation_config: dict.
        Arguments of permutation method.
    permutation_iters: int.
        Number of iterations of total variation optimization process.
    verbose: bool.
        If True, then information about model convertation process will be printed.
    """

    def __init__(
        self,
        init_from_discrete=True,
        fuse_bn=True,
        optimize_iters=0,
        start_lr=1e-2,
        permutation_config=None,
        parametrization_config=None,
        verbose=True,
    ):
        self.init_from_discrete = init_from_discrete
        self.fuse_bn = fuse_bn
        self.optimize_iters = optimize_iters
        self.start_lr = start_lr
        self.verbose = verbose
        self.rearranger = None

        if parametrization_config is None:
            self.parametrization_config = {'scale': 1, 'use_gridsample': True}
        else:
            self.parametrization_config = parametrization_config

        if self.init_from_discrete:
            permutation_class = NOptPermutation

            if permutation_config is None:
                permutation_config = {}

            if "class" in permutation_config:
                permutation_class = permutation_config.pop("class")

            self.rearranger = permutation_class(**permutation_config)

    def _rearrange(self, groups):
        """
        Rearranges the tensors in each group along continuous
        dimension to obtain continuous structure in tensors.

        Parameters
        ----------
        groups: List[RelatedGroup].
            List of related integral groups.
        """
        for i, group in enumerate(groups):
            params = list(group.params)
            feature_maps = group.tensors

            if self.verbose:
                print(f"Rearranging of group {i}")

            for parent in group.parents:
                start = 0

                for another_group in parent.subgroups:
                    if group is not another_group:
                        start += another_group.size
                    else:
                        break

                for p in parent.params:
                    params.append(
                        {
                            "name": p["name"],
                            "value": p["value"],
                            "dim": p["dim"],
                            "start_index": start,
                        }
                    )

            self.rearranger(params, feature_maps, group.size)

    def preprocess_model(
        self,
        model,
        example_input,
        continuous_dims,
        discrete_dims=None,
        integral_tracer_class=None,
        related_groups=None,
    ):
        """
        Builds dependency graph of the model, fuses BatchNorms
        and permutes tensor parameters along countinuous
        dimension to obtain smooth structure.

        Parameters
        ----------
        model: torch.nn.Module.
            Discrete neural network.
        example_input: torch.Tensor or List[int].
            Example input for the model.
        continuous_dims: Dict[str, List[int]].
            Dictionary with keys as names of parameters and values
            as lists of continuous dimensions of corresponding parameters.
        discrete_dims: Dict[str, List[int]].
            Dictionary with keys as names of parameters and values
            as lists of discrete dimensions of corresponding parameters.
        integral_tracer_class: IntegralTracer.
            Class inherited from IntegralTracer.

        Returns
        -------
        List[RelatedGroup].
            List of RelatedGroup objects.
        Dict[str, List[int]].
            Modified dictionary with continuous dimensions.
        """
        model.eval()
        if integral_tracer_class is None:
            integral_tracer_class = IntegralTracer

        if related_groups is None:
            tracer = integral_tracer_class(
                model,
                continuous_dims,
                discrete_dims,
            )
            tracer.build_groups(example_input)

            if self.fuse_bn:
                integral_convs = set()

                for name, _ in model.named_parameters():
                    if name in tracer.continuous_dims:
                        parent = get_parent_module(model, name)
                        dims = tracer.continuous_dims[name]

                        if isinstance(parent, nn.Conv2d) and 0 in dims:
                            integral_convs.add(get_parent_name(name)[0])

                fuse_batchnorm(
                    model, tracer.symbolic_trace(model), list(integral_convs)
                )

            tracer = integral_tracer_class(
                model,
                continuous_dims,
                discrete_dims,
            )
            related_groups = tracer.build_groups(example_input)

        if self.init_from_discrete and self.rearranger is not None:
            self._rearrange(related_groups)

        return related_groups, continuous_dims

    def __call__(
        self,
        model,
        example_input,
        continuous_dims,
        discrete_dims=None,
        integral_tracer_class=None,
        related_groups=None,
    ):
        """
        Parametrizes tensor parameters of the model
        and wraps the model into IntegralModel class.

        Parameters
        ----------
        model: torch.nn.Module.
            Discrete neural network.
        example_input: List[int] or torch.Tensor.
            Example input for the model.
        continuous_dims: Dict[str, List[int]].
            Dictionary with keys as names of parameters and values
            as lists of continuous dimensions of corresponding parameters.
        discrete_dims: Dict[str, List[int]].
            Dictionary with keys as names of parameters and values
            as lists of discrete dimensions of corresponding parameters.
        integral_tracer_class: IntegralTracer.
            Class inherited from IntegralTracer.
        related_groups: List[RelatedGroup].
            List of RelatedGroup objects collected outside.

        Returns
        -------
        IntegralModel.
            Model converted to integral form.
        """
        related_groups, continuous_dims = self.preprocess_model(
            model,
            example_input,
            continuous_dims,
            discrete_dims,
            integral_tracer_class,
            related_groups,
        )
        tmp_related = [None] * len(related_groups)
        for i, group in enumerate(related_groups):
            integral_group = IntegralGroup(group.size, group.base)
            integral_group.copy_attributes(group)
            group.toward = integral_group
            tmp_related[i] = integral_group

        for i, group in enumerate(related_groups):
            if group.partner is not None:
                group.toward.partner = group.partner.toward
                
        related_groups = tmp_related
        for group in related_groups:
            if group.grid is None:
                group.initialize_grids()
        visited_params = set()
        for group in related_groups:
            for param in group.params:
                _, name = get_parent_name(param["name"])
                parent = get_parent_module(model, param["name"])

                if param["name"] not in visited_params:
                    visited_params.add(param["name"])
                    dims = continuous_dims[param["name"]]
                    parametrization = self.build_parameterization(parent, name, dims)

                    if isinstance(parametrization, IWeights):
                        grids_list = []

                        for g in param["value"].related_groups:
                            if (
                                g is not None
                                and hasattr(g, "grid")
                                and g.grid is not None
                            ):
                                if g in related_groups:
                                    grids_list.append(g.grid)

                        delattr(param["value"], "related_groups")
                        grid = GridND(grids_list)
                        parametrization.grid = grid

                    parametrization.to(param["value"].device)
                    target = param["value"].detach().clone()
                    target.requires_grad = False
                    parametrize.register_parametrization(
                        parent, name, parametrization, unsafe=True
                    )

                    if self.init_from_discrete:
                        self._optimize_parameters(parent, param["name"], target)

                else:
                    parametrization = parent.parametrizations[name][0]

                param["function"] = parametrization

        integral_model = IntegralModel(model, related_groups)

        return integral_model

    def _optimize_parameters(self, module, name, target):
        """
        Optimize parametrization with Adam
        to approximate tensor attribute of given module.

        Parameters
        ----------
        module: torch.nn.Module.
            Layer of the model.
        name: str.
            Name of the parameter.
        target: torch.Tensor.
            Tensor to approximate.
        """
        module.train()
        _, attr = get_parent_name(name)
        criterion = torch.nn.MSELoss()
        opt = torch.optim.Adam(module.parameters(), lr=self.start_lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.optimize_iters // 5, gamma=0.2
        )

        if self.verbose:
            print(name)
            print(
                "loss before optimization: ",
                float(criterion(getattr(module, attr), target)),
            )

        for i in range(self.optimize_iters):
            weight = getattr(module, attr)
            loss = criterion(weight, target)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()

            if i == self.optimize_iters - 1 and self.verbose:
                print("loss after optimization: ", float(loss))


    def build_parameterization(self, module, name, dims):
        """
        Builds parametrization and quadrature objects
        for parameters of Conv2d, Conv1d or Linear

        Parameters
        ----------
        module: torhc.nn.Module.
            Layer of the model.
        name: str.
            Name of the parameter.
        dims: List[int].
            List of continuous dimensions of the parameter.
        scale: float.
            Parametrization size multiplier.
        gridsample: bool.
            If True then GridSampleWeights are used else InterpolationWeights

        Returns
        -------
        IntegralParameterization.
            Parametrization of the parameter.
        BaseIntegrationQuadrature.
            Quadrature object for the parameter.
        """
        quadrature = None
        func = None
        grid = None
        scale = self.parametrization_config['scale']
        use_gridsample = self.parametrization_config['use_gridsample']
        # if isinstance(parent, (nn.Linear, nn.Conv2d, nn.Conv1d)):

        if use_gridsample:
            parametrization_1d = GridSampleWeights1D
            parametrization_2d = GridSampleWeights2D
        else:
            parametrization_1d = InterpolationWeights1D
            parametrization_2d = InterpolationWeights2D

        if name == "weight" or (name=="bias" and len(getattr(module,name).shape) != 1):
            weight = getattr(module, name)
            cont_shape = [int(scale * weight.shape[d]) for d in dims]

            if 1 in dims and weight.shape[1] > 3:
                grid_indx = 0 if len(cont_shape) == 1 else 1
                quadrature = TrapezoidalQuadrature([1], [grid_indx])

            if weight.ndim > len(dims):
                discrete_shape = [
                    weight.shape[d] for d in range(weight.ndim) if d not in dims
                ]
            else:
                discrete_shape = None

            if len(cont_shape) == 2:
                func = parametrization_2d(grid, quadrature, cont_shape, discrete_shape)
            elif len(cont_shape) == 1:
                func = parametrization_1d(
                    grid, quadrature, cont_shape[0], discrete_shape, dims[0]
                )

        elif name == "bias":
            bias = getattr(module, name)
            cont_shape = int(scale * bias.shape[0])
            func = parametrization_1d(grid, quadrature, cont_shape)

        return func
