import torch
from math import sqrt

class RelatedGroup(torch.nn.Module):
    """
    Class for grouping tensors and parameters.
    Group is a collection of paris of tensor and it's dimension.
    Two parameter tensors are considered to be in the same group
    if they should have the same integration grid.
    Group can contain subgroups. This means that parent group's grid is a con
    catenation of subgroups grids.

    Parameters
    ----------
    size: int.
        Each tensor in the group should have the same size along certain dimension.
    """

    def __init__(self, size, typeo="normal", partner=None, factor=None):
        super(RelatedGroup, self).__init__()
        self.domain = None
        self.friends = None
        self.partner = partner
        self.factor = factor
        self.toward = None
        self.base = 1
        self._size = size
        self.subgroups = None
        self.parents = []
        self.params = []
        self.tensors = []
        self.operations = []
        self.typeo = typeo
    @property
    def size(self):
        if self.factor == "factor":
            if hasattr(self.partner, "grid"):
                return int(self.partner.grid_size() * self.factor)
            else:
                return int(self.partner.size * self.factor)
        elif self.typeo == "min":
            if hasattr(self.partner, "grid"):
                target = self.partner.grid_size()/3
            else:
                target = self.partner.size/3
            # tmp = int(sqrt(target))
            # max = 1
            # for i in range(tmp):
            #     if target % (i+1) == 0:
            #         max = i+1
            # return int(target / max)
            return int(target/16)
        elif self.typeo == "max":
            # if hasattr(self.partner, "grid"):
            #     target = self.partner.grid_size()
            # else:
            #     target = self.partner.size
            # tmp = int(sqrt(target))
            # max = 1
            # for i in range(tmp):
            #     if target % (i+1) == 0:
            #         max = i+1
            # return max
            return 16
        elif self.typeo == "concat":
            ret = 0
            for i in self.friends:
                if hasattr(i,"gird"):
                    ret += i.grid_size()
                else:
                    ret += i.size
            return ret
        else:
            return self._size
    def forward(self):
        pass

    def copy_attributes(self, group):
        self.domain = group.domain
        self.base = self.base 
        self._size = group.size
        self.subgroups = group.subgroups
        self.parents = group.parents
        self.params = group.params
        self.tensors = group.tensors
        self.operations = group.operations
        self.typeo = group.typeo
        self.factor = group.factor

        for parent in self.parents:
            if group in parent.subgroups:
                i = parent.subgroups.index(group)
                parent.subgroups[i] = self

        if self.subgroups is not None:
            for sub in self.subgroups:
                if group in sub.parents:
                    i = sub.parents.index(group)
                    sub.parents[i] = self

        for param in self.params:
            param["value"].related_groups[param["dim"]] = self

        for tensor in self.tensors:
            tensor["value"].related_groups[tensor["dim"]] = self

    def append_param(self, name, value, dim, operation=None):
        """
        Adds parameter tensor to the group.

        Parameters
        ----------
        name: str.
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.params.append(
            {"value": value, "name": name, "dim": dim, "operation": operation}
        )

    def append_tensor(self, value, dim, operation=None):
        """
        Adds tensor to the group.

        Parameters
        ----------
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.tensors.append({"value": value, "dim": dim, "operation": operation})

    def clear_params(self):
        self.params = []

    def clear_tensors(self):
        self.tensors = []

    def set_subgroups(self, groups):
        self.subgroups = groups

        for subgroup in self.subgroups:
            if subgroup is not None:
                subgroup.parents.append(self)

    def build_operations_set(self):
        """Builds set of operations in the group."""
        self.operations = set([t["operation"] for t in self.tensors])

    def count_parameters(self):
        ans = 0

        for p in self.params:
            ans += p["value"].numel()

        return ans

    def __str__(self):
        result = ""

        for p in self.params:
            result += p["name"] + ": " + str(p["dim"]) + "\n"

        return result

    @staticmethod
    def append_to_groups(tensor, operation=None):
        attr_name = "related_groups"

        if hasattr(tensor, attr_name):
            for i, g in enumerate(getattr(tensor, attr_name)):
                if g is not None:
                    g.append_tensor(tensor, i, operation)

# class FactorRelatedGroup(torch.nn.Module):
#     """
#     Class for grouping tensors and parameters.
#     Group is a collection of paris of tensor and it's dimension.
#     Two parameter tensors are considered to be in the same group
#     if they should have the same integration grid.
#     Group can contain subgroups. This means that parent group's grid is a con
#     catenation of subgroups grids.

#     Parameters
#     ----------
#     size: int.
#         Each tensor in the group should have the same size along certain dimension.
#     """

#     def __init__(self, partner, factor):
#         super(FactorRelatedGroup, self).__init__()
#         self.toward = None
#         self.partner = partner
#         self.factor = factor
#         self.subgroups = None
#         self.parents = []
#         self.params = []
#         self.tensors = []
#         self.operations = []

#     def forward(self):
#         pass

#     def copy_attributes(self, group):
#         self.subgroups = group.subgroups
#         self.parents = group.parents
#         self.params = group.params
#         self.tensors = group.tensors
#         self.operations = group.operations

#         for parent in self.parents:
#             if group in parent.subgroups:
#                 i = parent.subgroups.index(group)
#                 parent.subgroups[i] = self

#         if self.subgroups is not None:
#             for sub in self.subgroups:
#                 if group in sub.parents:
#                     i = sub.parents.index(group)
#                     sub.parents[i] = self

#         for param in self.params:
#             param["value"].related_groups[param["dim"]] = self

#         for tensor in self.tensors:
#             tensor["value"].related_groups[tensor["dim"]] = self

#     def append_param(self, name, value, dim, operation=None):
#         """
#         Adds parameter tensor to the group.

#         Parameters
#         ----------
#         name: str.
#         value: torch.Tensor.
#         dim: int.
#         operation: str.
#         """
#         self.params.append(
#             {"value": value, "name": name, "dim": dim, "operation": operation}
#         )

#     def append_tensor(self, value, dim, operation=None):
#         """
#         Adds tensor to the group.

#         Parameters
#         ----------
#         value: torch.Tensor.
#         dim: int.
#         operation: str.
#         """
#         self.tensors.append({"value": value, "dim": dim, "operation": operation})

#     def clear_params(self):
#         self.params = []

#     def clear_tensors(self):
#         self.tensors = []

#     def set_subgroups(self, groups):
#         self.subgroups = groups

#         for subgroup in self.subgroups:
#             if subgroup is not None:
#                 subgroup.parents.append(self)

#     def build_operations_set(self):
#         """Builds set of operations in the group."""
#         self.operations = set([t["operation"] for t in self.tensors])

#     def count_parameters(self):
#         ans = 0

#         for p in self.params:
#             ans += p["value"].numel()

#         return ans

#     def __str__(self):
#         result = ""

#         for p in self.params:
#             result += p["name"] + ": " + str(p["dim"]) + "\n"

#         return result

#     @staticmethod
#     def append_to_groups(tensor, operation=None):
#         attr_name = "related_groups"

#         if hasattr(tensor, attr_name):
#             for i, g in enumerate(getattr(tensor, attr_name)):
#                 if g is not None:
#                     g.append_tensor(tensor, i, operation)

# class CoMaxRelatedGroup(torch.nn.Module):
#     """
#     self.size := max(A) where A = {a | (partner.size % a == 0) and a >= sqrt(partner.size)}
#     """

#     def __init__(self, partner):
#         super(CoMaxRelatedGroup, self).__init__()
#         self.toward = None
#         self.partner = partner
#         self.SOUL = 1
#         self.subgroups = None
#         self.parents = []
#         self.params = []
#         self.tensors = []
#         self.operations = []

#     @property
#     def size(self):
#         if hasattr(self.partner, "grid"):
#             target = self.partner.grid_size()
#         else:
#             target = self.partner.size
#         # tmp = int(sqrt(target))
#         # max = 1
#         # for i in range(tmp):
#         #     if target % (i+1) == 0:
#         #         max = i+1
#         return int(target / 16)
#     def forward(self):
#         pass

#     def copy_attributes(self, group):
#         self.subgroups = group.subgroups
#         self.parents = group.parents
#         self.params = group.params
#         self.tensors = group.tensors
#         self.operations = group.operations

#         for parent in self.parents:
#             if group in parent.subgroups:
#                 i = parent.subgroups.index(group)
#                 parent.subgroups[i] = self

#         if self.subgroups is not None:
#             for sub in self.subgroups:
#                 if group in sub.parents:
#                     i = sub.parents.index(group)
#                     sub.parents[i] = self

#         for param in self.params:
#             param["value"].related_groups[param["dim"]] = self

#         for tensor in self.tensors:
#             tensor["value"].related_groups[tensor["dim"]] = self

#     def append_param(self, name, value, dim, operation=None):
#         """
#         Adds parameter tensor to the group.

#         Parameters
#         ----------
#         name: str.
#         value: torch.Tensor.
#         dim: int.
#         operation: str.
#         """
#         self.params.append(
#             {"value": value, "name": name, "dim": dim, "operation": operation}
#         )

#     def append_tensor(self, value, dim, operation=None):
#         """
#         Adds tensor to the group.

#         Parameters
#         ----------
#         value: torch.Tensor.
#         dim: int.
#         operation: str.
#         """
#         self.tensors.append({"value": value, "dim": dim, "operation": operation})

#     def clear_params(self):
#         self.params = []

#     def clear_tensors(self):
#         self.tensors = []

#     def set_subgroups(self, groups):
#         self.subgroups = groups

#         for subgroup in self.subgroups:
#             if subgroup is not None:
#                 subgroup.parents.append(self)

#     def build_operations_set(self):
#         """Builds set of operations in the group."""
#         self.operations = set([t["operation"] for t in self.tensors])

#     def count_parameters(self):
#         ans = 0

#         for p in self.params:
#             ans += p["value"].numel()

#         return ans

#     def __str__(self):
#         result = ""

#         for p in self.params:
#             result += p["name"] + ": " + str(p["dim"]) + "\n"

#         return result

#     @staticmethod
#     def append_to_groups(tensor, operation=None):
#         attr_name = "related_groups"

#         if hasattr(tensor, attr_name):
#             for i, g in enumerate(getattr(tensor, attr_name)):
#                 if g is not None:
#                     g.append_tensor(tensor, i, operation)

# class CoMinRelatedGroup(torch.nn.Module):
#     """
#     self.size := max(A) where A = {a | (partner.size % a == 0) and a >= sqrt(partner.size)}
#     """

#     def __init__(self, partner):
#         super(CoMinRelatedGroup, self).__init__()
#         self.toward = None
#         self.partner = partner
#         self.SOUL = 0
#         self.subgroups = None
#         self.parents = []
#         self.params = []
#         self.tensors = []
#         self.operations = []

#     @property
#     def size(self):
#         # if hasattr(self.partner, "grid"):
#         #     target = self.partner.grid.related_size / 3
#         # else:
#         #     target = self.partner.size / 3
#         # tmp = int(sqrt(target))
#         # max = 1
#         # for i in range(tmp):
#         #     if target % (i+1) == 0:
#         #         max = i+1
#         return 16
#     def forward(self):
#         pass

#     def copy_attributes(self, group):
#         self.subgroups = group.subgroups
#         self.parents = group.parents
#         self.params = group.params
#         self.tensors = group.tensors
#         self.operations = group.operations

#         for parent in self.parents:
#             if group in parent.subgroups:
#                 i = parent.subgroups.index(group)
#                 parent.subgroups[i] = self

#         if self.subgroups is not None:
#             for sub in self.subgroups:
#                 if group in sub.parents:
#                     i = sub.parents.index(group)
#                     sub.parents[i] = self

#         for param in self.params:
#             param["value"].related_groups[param["dim"]] = self

#         for tensor in self.tensors:
#             tensor["value"].related_groups[tensor["dim"]] = self

#     def append_param(self, name, value, dim, operation=None):
#         """
#         Adds parameter tensor to the group.

#         Parameters
#         ----------
#         name: str.
#         value: torch.Tensor.
#         dim: int.
#         operation: str.
#         """
#         self.params.append(
#             {"value": value, "name": name, "dim": dim, "operation": operation}
#         )

#     def append_tensor(self, value, dim, operation=None):
#         """
#         Adds tensor to the group.

#         Parameters
#         ----------
#         value: torch.Tensor.
#         dim: int.
#         operation: str.
#         """
#         self.tensors.append({"value": value, "dim": dim, "operation": operation})

#     def clear_params(self):
#         self.params = []

#     def clear_tensors(self):
#         self.tensors = []

#     def set_subgroups(self, groups):
#         self.subgroups = groups

#         for subgroup in self.subgroups:
#             if subgroup is not None:
#                 subgroup.parents.append(self)

#     def build_operations_set(self):
#         """Builds set of operations in the group."""
#         self.operations = set([t["operation"] for t in self.tensors])

#     def count_parameters(self):
#         ans = 0

#         for p in self.params:
#             ans += p["value"].numel()

#         return ans

#     def __str__(self):
#         result = ""

#         for p in self.params:
#             result += p["name"] + ": " + str(p["dim"]) + "\n"

#         return result

#     @staticmethod
#     def append_to_groups(tensor, operation=None):
#         attr_name = "related_groups"

#         if hasattr(tensor, attr_name):
#             for i, g in enumerate(getattr(tensor, attr_name)):
#                 if g is not None:
#                     g.append_tensor(tensor, i, operation)