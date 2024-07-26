import torch
from .grid import RandomLinspace, UniformDistribution, FactorUniformDistribution, CompositeGrid1D, MinRelatedDistribution, MaxRelatedDistribution
from .graph import RelatedGroup

class IntegralGroup(RelatedGroup):
    """ """

    def __init__(self, size, base, typeo="normal", factor=None, partner=None):
        super(RelatedGroup, self).__init__()
        self.rate = 1
        self.ori_size = size
        self.stop = False
        self.factor = factor
        self.domain = None
        self.friends = None
        self.partner = partner
        self._size = size
        self.base = base
        self.subgroups = None
        self.parents = []
        self.grid = None
        self.params = []
        self.tensors = []
        self.operations = []
        self.typeo = typeo
        self.vis = False
    def set_vis(self):
        if self.subgroups is not None:
            for i in self.subgroups:
                i.set_vis()
        self.vis = True
    def forward(self):
        self.set_vis()
        self.grid.generate_grid()

    def grid_size(self):
        """Returns size of the grid."""
        return self.grid.size()

    def clear(self, new_grid=None):
        """Resets grid and removes cached values."""
        for param_dict in self.params:
            function = param_dict["function"]
            dim = list(function.grid).index(self.grid)
            grid = new_grid if new_grid is not None else self.grid
            function.grid.reset_grid(dim, grid)
            function.clear()

    def initialize_grids(self):
        """Sets default RandomLinspace grid."""
        if self.grid is None:
            if self.partner is not None:
                if self.partner.grid is None:
                    self.partner.initialize_grids()
            if self.subgroups is not None:
                for subgroup in self.subgroups:
                    if subgroup.grid is None:
                        subgroup.initialize_grids()

                self.grid = CompositeGrid1D([sub.grid for sub in self.subgroups])
            else:
                if self.typeo == "factor":
                    distrib = FactorUniformDistribution(self.partner, self.factor)
                elif self.typeo == "max":
                    distrib = MaxRelatedDistribution(self.partner)
                elif self.typeo == "min":
                    distrib = MinRelatedDistribution(self.partner)
                else:
                    if self.domain is None:
                        distrib = UniformDistribution(self.size, self.size, self.base)
                    else:
                        distrib = UniformDistribution(self.size, self.size, self.base, domain=self.domain)
                self.grid = RandomLinspace(distrib)

    def reset_grid(self, new_grid):
        """
        Set new integration grid for the group.

        Parameters
        ----------
        new_grid: IntegralGrid.
        """
        self.clear(new_grid)

        for parent in self.parents:
            parent.reset_child_grid(self, new_grid)

        self.grid = new_grid

    def reset_child_grid(self, child, new_grid):
        """Sets new integration grid for given child of the group."""
        i = self.subgroups.index(child)
        self.grid.reset_grid(i, new_grid)
        self.clear()

    def resize(self, new_size):
        """If grid supports resizing, resizes it."""
        self._size = new_size
        if hasattr(self.grid, "resize"):
            self.grid.resize(new_size)

        self.clear()

        for parent in self.parents:
            parent.clear()

    def reset_distribution(self, distribution):
        """Sets new distribution for the group."""
        if self.typeo == "normal" and hasattr(self.grid, "distribution"):
            self.grid.distribution = distribution
# class FactorIntegralGroup(FactorRelatedGroup):
#     """ 
#     self.size = partner.size * factor
#     """
#     def __init__(self, partner, factor):
#         super(FactorRelatedGroup, self).__init__()
#         self.factor = factor
#         self.partner = partner
#         self.subgroups = None
#         self.parents = []
#         self.grid = None
#         self.params = []
#         self.tensors = []
#         self.operations = []

#     def forward(self):
#         self.grid.distribution.father = self.partner
#         self.grid.generate_grid()

#     def grid_size(self):
#         """Returns size of the grid."""
#         return self.grid.size()

#     def clear(self, new_grid=None):
#         """Resets grid and removes cached values."""
#         for param_dict in self.params:
#             function = param_dict["function"]
#             dim = list(function.grid).index(self.grid)
#             grid = new_grid if new_grid is not None else self.grid
#             function.grid.reset_grid(dim, grid)
#             function.clear()

#     def initialize_grids(self):
#         """Sets default RandomLinspace grid."""
#         if self.grid is None:
#             if self.partner.grid is None:
#                 self.partner.initialize_grids()
#             if self.subgroups is not None:
#                 for subgroup in self.subgroups:
#                     if subgroup.grid is None:
#                         subgroup.initialize_grids()

#                 self.grid = CompositeGrid1D([sub.grid for sub in self.subgroups])
#             else:
#                 distrib = FactorUniformDistribution(self.partner, self.factor)
#                 self.grid = RandomLinspace(distrib)

#     def reset_grid(self, new_grid):
#         """
#         Set new integration grid for the group.

#         Parameters
#         ----------
#         new_grid: IntegralGrid.
#         """
#         self.clear(new_grid)

#         for parent in self.parents:
#             parent.reset_child_grid(self, new_grid)

#         self.grid = new_grid

#     def reset_child_grid(self, child, new_grid):
#         """Sets new integration grid for given child of the group."""
#         i = self.subgroups.index(child)
#         self.grid.reset_grid(i, new_grid)
#         self.clear()

#     def resize(self, new_size):
#         """If grid supports resizing, resizes it."""
#         if hasattr(self.grid, "resize"):
#             self.grid.resize(new_size)

#         self.clear()

#         for parent in self.parents:
#             parent.clear()

#     def reset_distribution(self, distri):
#         pass

# class CoMaxIntegralGroup(CoMaxRelatedGroup):
#     """ """

#     def __init__(self, partner):
#         super(CoMaxRelatedGroup, self).__init__()
#         self.partner = partner
#         self.SOUL = 1
#         self.subgroups = None
#         self.parents = []
#         self.grid = None
#         self.params = []
#         self.tensors = []
#         self.operations = []

#     def forward(self):
#         self.grid.distribution.father = self.partner
#         self.grid.generate_grid()

#     def grid_size(self):
#         """Returns size of the grid."""
#         return self.grid.size()

#     def clear(self, new_grid=None):
#         """Resets grid and removes cached values."""
#         for param_dict in self.params:
#             function = param_dict["function"]
#             dim = list(function.grid).index(self.grid)
#             grid = new_grid if new_grid is not None else self.grid
#             function.grid.reset_grid(dim, grid)
#             function.clear()

#     def initialize_grids(self):
#         """Sets default RandomLinspace grid."""
#         if self.grid is None:
#             if self.partner.grid is None:
#                 self.partner.initialize_grids()            
#             if self.subgroups is not None:
#                 for subgroup in self.subgroups:
#                     if subgroup.grid is None:
#                         subgroup.initialize_grids()

#                 self.grid = CompositeGrid1D([sub.grid for sub in self.subgroups])
#             else:
#                 distrib = MaxRelatedDistribution(self.partner)
#                 self.grid = RandomLinspace(distrib)

#     def reset_grid(self, new_grid):
#         """
#         Set new integration grid for the group.

#         Parameters
#         ----------
#         new_grid: IntegralGrid.
#         """
#         self.clear(new_grid)

#         for parent in self.parents:
#             parent.reset_child_grid(self, new_grid)

#         self.grid = new_grid

#     def reset_child_grid(self, child, new_grid):
#         """Sets new integration grid for given child of the group."""
#         i = self.subgroups.index(child)
#         self.grid.reset_grid(i, new_grid)
#         self.clear()

#     def resize(self, new_size):
#         """If grid supports resizing, resizes it."""
#         if hasattr(self.grid, "resize"):
#             self.grid.resize(new_size)

#         self.clear()

#         for parent in self.parents:
#             parent.clear()

#     def reset_distribution(self, distribution):
#         pass

# class CoMinIntegralGroup(CoMinRelatedGroup):
#     """ """

#     def __init__(self, partner):
#         super(CoMinRelatedGroup, self).__init__()
#         self.partner = partner
#         self.SOUL = 0
#         self.subgroups = None
#         self.parents = []
#         self.grid = None
#         self.params = []
#         self.tensors = []
#         self.operations = []

#     def forward(self):
#         self.grid.distribution.father = self.partner
#         self.grid.generate_grid()

#     def grid_size(self):
#         """Returns size of the grid."""
#         return self.grid.size()

#     def clear(self, new_grid=None):
#         """Resets grid and removes cached values."""
#         for param_dict in self.params:
#             function = param_dict["function"]
#             dim = list(function.grid).index(self.grid)
#             grid = new_grid if new_grid is not None else self.grid
#             function.grid.reset_grid(dim, grid)
#             function.clear()

#     def initialize_grids(self):
#         """Sets default RandomLinspace grid."""
#         if self.grid is None:
#             if self.partner.grid is None:
#                 self.partner.initialize_grids()
#             if self.subgroups is not None:
#                 for subgroup in self.subgroups:
#                     if subgroup.grid is None:
#                         subgroup.initialize_grids()

#                 self.grid = CompositeGrid1D([sub.grid for sub in self.subgroups])
#             else:
#                 distrib = MinRelatedDistribution(self.partner)
#                 self.grid = RandomLinspace(distrib)

#     def reset_grid(self, new_grid):
#         """
#         Set new integration grid for the group.

#         Parameters
#         ----------
#         new_grid: IntegralGrid.
#         """
#         self.clear(new_grid)

#         for parent in self.parents:
#             parent.reset_child_grid(self, new_grid)

#         self.grid = new_grid

#     def reset_child_grid(self, child, new_grid):
#         """Sets new integration grid for given child of the group."""
#         i = self.subgroups.index(child)
#         self.grid.reset_grid(i, new_grid)
#         self.clear()

#     def resize(self, new_size):
#         """If grid supports resizing, resizes it."""
#         if hasattr(self.grid, "resize"):
#             self.grid.resize(new_size)

#         self.clear()

#         for parent in self.parents:
#             parent.clear()

#     def reset_distribution(self, distribution):
#         pass