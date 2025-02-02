import torch
import random
from scipy.special import roots_legendre
from math import log,exp,sqrt

class Distribution:
    """
    Base class for grid size distribution.

    Attributes
    ----------
    min_val: int.
        Minimal possible random value.
    max_val: int.
        Maximal possible random value.
    """

    def __init__(self, min_val, max_val):
        self.min_val = int(min_val)
        self.max_val = int(max_val)

    def sample(self):
        """Samples random integer number from distribution."""
        raise NotImplementedError("Implement this method in derived class.")


class UniformDistribution(Distribution):
    def __init__(self, min_val, max_val, factor, domain=None):
        self.factor = int(factor)
        self.domain = domain
        super().__init__(min_val, max_val)

    def sample(self):
        if self.domain is None or (self.min_val == self.max_val):
            # out = random.normalvariate(0, 1/3*(self.max_val/self.factor - self.min_val/self.factor))
            # out = int(abs(out))*self.factor + self.min_val
            # if (out < self.max_val) and (self.min_val < out):
            #     return out
            # else:
            return random.randint(int(self.min_val/self.factor), int(self.max_val/self.factor)) * self.factor
        else:
            return self.domain[random.randint(0, len(self.domain)-1)]

class FactorUniformDistribution(Distribution):
    def __init__(self, father, factor):
        self.father = father
        self.factor = factor
        if hasattr(self.father, "grid"):
            related_size = self.father.grid_size()
        else:
            related_size = self.father.size
        super().__init__(int(related_size * self.factor), int(related_size * self.factor))
    def sample(self):
        if hasattr(self.father, "grid"):
            related_size = self.father.grid_size()
        else:
            related_size = self.father.size
        return int(related_size * self.factor)
class MaxRelatedDistribution(Distribution):
    def __init__(self, father):
        self.father = father
        # if hasattr(self.father, "grid"):
        #     target = self.father.grid_size()
        # else:
        #     target = self.father.size
        # tmp = int(sqrt(target))
        # max = 1
        # for i in range(tmp):
        #     if target % (i+1) == 0:
        #         max = i+1
        # super().__init__(max, max)
        super().__init__(16, 16)
    def sample(self):
        # if hasattr(self.father, "grid"):
        #     target = self.father.grid_size()
        # else:
        #     target = self.father.size
        # tmp = int(sqrt(target))
        # max = 1
        # for i in range(tmp):
        #     if target % (i+1) == 0:
        #         max = i+1
        # return max
        return 16
class MinRelatedDistribution(Distribution):
    def __init__(self, father):
        self.father = father
        if hasattr(self.father, "grid"):
            target = self.father.grid_size()/3
        else:
            target = self.father.size/3
        # tmp = int(sqrt(target))
        # max = 1
        # for i in range(tmp):
        #     if target % (i+1) == 0:
        #         max = i+1
        # super().__init__(int(target/max), int(target/max))
        super().__init__(int(target/16),int(target/16))
    def sample(self):
        if hasattr(self.father, "grid"):
            target = self.father.grid_size()/3
        else:
            target = self.father.size/3
        # tmp = int(sqrt(target))
        # max = 1
        # for i in range(tmp):
        #     if target % (i+1) == 0:
        #         max = i+1
        return int(target/16)  
class NormalDistribution(Distribution):
    def __init__(self, min_val, max_val):
        super(NormalDistribution, self).__init__(min_val, max_val)

    def sample(self):
        out = random.normalvariate(0, 0.5 * (self.max_val - self.min_val))
        out = self.max_val - int(abs(out))

        if out < self.min_val:
            out = random.randint(self.min_val, self.max_val)

        return out


class IGrid(torch.nn.Module):
    """Base Grid class."""

    def __init__(self):
        super(IGrid, self).__init__()
        self.curr_grid = None
        self.eval_size = None

    def forward(self):
        """
        Performs forward pass. Generates new grid if
        last generated grid is not saved, else returns saved one.

        Returns
        -------
        torch.Tensor.
            Generated grid points.
        """
        if self.curr_grid is None:
            out = self.generate_grid()
        else:
            out = self.curr_grid

        return out

    def ndim(self):
        """Returns dimensionality of grid object."""
        return 1

    def size(self):
        return self.eval_size

    def generate_grid(self):
        """Samples new grid points."""
        raise NotImplementedError("Implement this method in derived class.")


class ConstantGrid1D(IGrid):
    """
    Class implements IGrid interface for fixed grid.

    Parameters
    ----------
    init_value: torch.Tensor.
    """

    def __init__(self, init_value):
        super(ConstantGrid1D, self).__init__()
        self.curr_grid = init_value

    def generate_grid(self):
        return self.curr_grid


class TrainableGrid1D(IGrid):
    """Grid with TrainablePartition.

    Parameters
    ----------
    size: int.
    init_value: torch.Tensor.
    """

    def __init__(self, size, init_value=None):
        super(TrainableGrid1D, self).__init__()
        self.eval_size = size
        self.curr_grid = torch.nn.Parameter(torch.linspace(-1, 1, size))
        if init_value is not None:
            assert size == init_value.shape[0]
            self.curr_grid.data = init_value

    def generate_grid(self):
        return self.curr_grid


class L1Grid1D(IGrid):
    def __init__(self, group, size):
        super().__init__()
        indices = self.get_indices(group, size).cpu()
        self.curr_grid = torch.linspace(-1, 1, group.size).index_select(0, indices)
        self.curr_grid = self.curr_grid.sort().values

    def generate_grid(self):
        return self.generate_grid

    def get_indices(self, group, size):
        device = group.params[0]["value"].device
        channels_importance = torch.zeros(group.size, device=device)

        for param in group.params:
            if "bias" not in param["name"]:
                tensor = param["value"]
                tensor = param["function"](tensor)
                dim = param["dim"]
                tensor = tensor.transpose(0, dim).reshape(group.size, -1)
                mean = tensor.abs().mean(dim=1)
                channels_importance += mean

        return torch.argsort(channels_importance)[:size]


class MultiTrainableGrid1D(IGrid):
    def __init__(self, full_grid, index, num_grids):
        super(MultiTrainableGrid1D, self).__init__()
        self.curr_grid = None
        self.num_grids = num_grids
        self.full_grid = full_grid
        self.index = index
        self.generate_grid()

    def generate_grid(self):
        grid_len = 1.0 / self.num_grids
        start = self.index * grid_len
        end = start + grid_len
        grid = self.full_grid[(self.full_grid >= start) & (self.full_grid < end)]
        self.curr_grid = 2 * (grid - start) / grid_len - 1.0

        return self.curr_grid


class TrainableDeltasGrid1D(IGrid):
    """Grid with TrainablePartition parametrized with deltas.

    Parameters
    ----------
    size: int.
    """

    def __init__(self, size):
        super(TrainableDeltasGrid1D, self).__init__()
        self.eval_size = size
        self.deltas = torch.nn.Parameter(torch.zeros(size - 1))
        self.curr_grid = None

    def generate_grid(self):
        self.curr_grid = torch.cumsum(self.deltas.abs(), dim=0)
        self.curr_grid = torch.cat([torch.zeros(1), self.curr_grid])
        self.curr_grid = self.curr_grid * 2 - 1

        return self.curr_grid


class RandomLinspace(IGrid):
    """
    Grid which generates random sized tensor each time,
    when generate_grid method is called.
    Size of tensor is sampled from ``size_distribution``.

    Parameters
    ----------
    size_distribution: Distribution.
    noise_std: float.
    """

    def __init__(self, size_distribution, noise_std=0):
        super(RandomLinspace, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        self.noise_std = noise_std
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size
        self.curr_grid = torch.linspace(-1, 1, size)
        
        if self.noise_std > 0:
            noise = torch.normal(torch.zeros(size), self.noise_std * torch.ones(size))
            self.curr_grid = self.curr_grid + noise

        return self.curr_grid

    def resize(self, new_size):
        """Set new value for evaluation size."""
        self.eval_size = new_size
        self.generate_grid()

    def size(self):
        return len(self.curr_grid)
class RandomLegendreGrid(RandomLinspace):
    def __init__(self, size_distribution):
        super(RandomLinspace, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size

        self.curr_grid, _ = roots_legendre(size)
        self.curr_grid = torch.tensor(self.curr_grid, dtype=torch.float32)

        return self.curr_grid
class CompositeGrid1D(IGrid):
    """Grid which consist of concatenated IGrid objects."""

    def __init__(self, grids):
        super(CompositeGrid1D, self).__init__()
        self.grids = torch.nn.ModuleList(grids)
        size = self.size()
        self.proportions = [(grid.size() - 1) / (size - 1) for grid in grids]
        self.generate_grid()

    def reset_grid(self, index, new_grid):
        self.grids[index] = new_grid
        self.generate_grid()

    def generate_grid(self):
        g_list = []
        start = 0.0
        h = 1 / (self.size() - 1)
        device = None

        for i, grid in enumerate(self.grids):
            g = grid.generate_grid()
            device = g.device if device is None else device
            g = (g + 1.0) / 2.0
            g = start + g * self.proportions[i]
            g_list.append(g.to(device))
            start += self.proportions[i] + h

        self.curr_grid = 2.0 * torch.cat(g_list) - 1.0

        return self.curr_grid

    def size(self):
        return sum([g.size() for g in self.grids])


class GridND(IGrid):
    """N-dimensional grid, each dimension of which is an object of type IGrid."""

    def __init__(self, grid_objects):
        super(GridND, self).__init__()
        self.grid_objects = torch.nn.ModuleList(grid_objects)

        self.generate_grid()

    def ndim(self):
        """Returns dimensionality of grid object."""
        return sum([grid.ndim() for grid in self.grid_objects])

    def reset_grid(self, dim, new_grid):
        """Replaces grid at given index."""
        self.grid_objects[dim] = new_grid
        self.generate_grid()

    def generate_grid(self):
        self.curr_grid = [grid.generate_grid() for grid in self.grid_objects]

        return self.curr_grid

    def forward(self):
        self.curr_grid = [grid() for grid in self.grid_objects]

        return self.curr_grid

    def __iter__(self):
        return iter(self.grid_objects)
