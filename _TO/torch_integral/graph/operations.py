import operator
import torch
from .group import RelatedGroup
import torch.nn as nn

def merge_groups(x, x_dim, y, y_dim):
    """Merges two groups of tensors ``x`` and `yy`` with indices ``x_dim`` and ``y_dim``."""
    if type(x) in (int, float):
        x = torch.tensor(x)
    if type(y) in (int, float):
        y = torch.tensor(y)
    if not hasattr(x, "related_groups"):
        x.related_groups = [None for _ in range(x.ndim)]
    if not hasattr(y, "related_groups"):
        y.related_groups = [None for _ in range(y.ndim)]
    if y.related_groups[y_dim] is not None:
        x, x_dim, y, y_dim = y, y_dim, x, x_dim
    if x.related_groups[x_dim] is not None:
        if y.related_groups[y_dim] is not None:
            if x.related_groups[x_dim].typeo == "concat":
                y.related_groups[y_dim].friends = x.related_groups[x_dim].friends
                y.related_groups[y_dim].typeo = "concat"
            elif y.related_groups[y_dim].typeo == "concat":
                x.related_groups[x_dim].friends = y.related_groups[y_dim].friends
                x.related_groups[x_dim].typeo = "concat"
            if len(y.related_groups[y_dim].parents) > 0:
                x, x_dim, y, y_dim = y, y_dim, x, x_dim

            if y.related_groups[y_dim].subgroups is not None:
                x, x_dim, y, y_dim = y, y_dim, x, x_dim
            
            if x.related_groups[x_dim] is not y.related_groups[y_dim]:
                y.related_groups[y_dim].toward = x.related_groups[x_dim]
                #need to change for lcm(x.base, y.base)
                if x.related_groups[x_dim].base < y.related_groups[y_dim].base:
                    x.related_groups[x_dim].base = y.related_groups[y_dim].base
                else:
                    y.related_groups[y_dim].base = x.related_groups[x_dim].base
                
                for param in y.related_groups[y_dim].params:
                    dim = param["dim"]
                    t = param["value"]

                    if t is not y:
                        t.related_groups[dim] = x.related_groups[x_dim]

                x.related_groups[x_dim].params.extend(y.related_groups[y_dim].params)
                y.related_groups[y_dim].clear_params()

                for tensor in y.related_groups[y_dim].tensors:
                    dim = tensor["dim"]
                    t = tensor["value"]

                    if t is not y:
                        t.related_groups[dim] = x.related_groups[x_dim]

                x.related_groups[x_dim].tensors.extend(y.related_groups[y_dim].tensors)
                y.related_groups[y_dim].clear_tensors()

        y.related_groups[y_dim] = x.related_groups[x_dim]    

def neutral_hook(module, input, output):
    if hasattr(input[0], "related_groups"):
        output.related_groups = input[0].related_groups
        RelatedGroup.append_to_groups(output, "neutral")


def conv_linear_hook(module, input, output):
    weight = module.weight
    bias = module.bias
    if bias is not None:
        merge_groups(bias, 0, weight, 0)

    merge_groups(weight, 1, input[0], 1)
    merge_groups(output, 1, weight, 0)
    RelatedGroup.append_to_groups(output, "conv_linear")


def conv_transposed_hook(module, input, output):
    weight = module.weight
    bias = module.bias

    if bias is not None:
        merge_groups(bias, 0, weight, 1)

    merge_groups(weight, 0, input[0], 1)
    merge_groups(output, 1, weight, 1)
    RelatedGroup.append_to_groups(output, "conv_transposed")

def layer_norm_hook(module, inp, out):
    weight = module.weight
    bias = module.bias
    merge_groups(bias, 0, weight, 0)
    merge_groups(bias, 1, weight, 1)
    merge_groups(inp[0], 2, weight, 0)
    merge_groups(inp[0], 3, weight, 1)
    # merge_groups(weight, 0, inp[0], 2)
    # merge_groups(weight, 1, inp[0], 3)
    merge_groups(out, 2, weight, 0)
    merge_groups(out, 3, weight, 1)
    RelatedGroup.append_to_groups(out, "layer_norm")

def galerkin_hook(module, inp, out):
    qkv_proj_weight = module.qkv_proj.weight
    qkv_proj_bias = module.qkv_proj.bias
    kln_weight = module.kln.weight
    kln_bias = module.kln.bias
    vln_weight = module.vln.weight
    vln_bias = module.vln.bias
    core_proj_weight = module.core_proj.weight
    core_proj_bias = module.core_proj.bias
    #inp as [1,256,128,128]
    #out as [1,256,128,128]
    merge_groups(inp[0], 1, qkv_proj_weight, 1)
    # qkv_proj_weight.related_groups[1].domain = [64, 72, 80, 81, 88, 90, 96, 99, 100, 104, 108, 110, 112, 117, 120, 121, 126, 128, 130, 132, 135, 140, 143, 144, 150, 154, 156, 160, 165, 168, 169, 176, 180, 182, 192, 195, 196, 208, 210, 224, 225, 240, 256]
    qkv_proj_weight.related_groups[1].base = 16

    # qkv_proj_weight.related_groups[0].typeo = "factor"
    # qkv_proj_weight.related_groups[0].partner = qkv_proj_weight.related_groups[1]
    qkv_proj_weight.related_groups[0].base = 48
    # qkv_proj_bias.related_groups[0].typeo = "factor"
    # qkv_proj_bias.related_groups[0].partner = qkv_proj_weight.related_groups[1]
    qkv_proj_bias.related_groups[0].base = 48
    merge_groups(qkv_proj_bias, 0, qkv_proj_weight, 0)
    RelatedGroup.append_to_groups(qkv_proj_weight, "galerkin attention")
    #qkv_proj out as [B,768,128,128]
    kln_bias.related_groups[0].typeo = "max"
    kln_bias.related_groups[1].typeo = "min"
    kln_bias.related_groups[0].partner = qkv_proj_weight.related_groups[0]
    kln_bias.related_groups[1].partner = qkv_proj_weight.related_groups[0]
    kln_weight.related_groups[0].typeo = "max"
    kln_weight.related_groups[1].typeo = "min"
    kln_weight.related_groups[0].partner = qkv_proj_weight.related_groups[0]
    kln_weight.related_groups[1].partner = qkv_proj_weight.related_groups[0]
    vln_bias.related_groups[0].typeo = "max"
    vln_bias.related_groups[1].typeo = "min"
    vln_bias.related_groups[0].partner = qkv_proj_weight.related_groups[0]
    vln_bias.related_groups[1].partner = qkv_proj_weight.related_groups[0]
    vln_weight.related_groups[0].typeo = "max"
    vln_weight.related_groups[1].typeo = "min"
    vln_weight.related_groups[0].partner = qkv_proj_weight.related_groups[0]
    vln_weight.related_groups[1].partner = qkv_proj_weight.related_groups[0]
    merge_groups(kln_bias, 0, kln_weight, 0)
    merge_groups(vln_bias, 0, vln_weight, 0)
    merge_groups(vln_weight, 0, kln_weight, 0)
    

    merge_groups(kln_bias, 1, kln_weight, 1)
    merge_groups(vln_bias, 1, vln_weight, 1)
    merge_groups(vln_weight, 1, kln_weight, 1)

    RelatedGroup.append_to_groups(kln_weight, "galerkin attention")
    merge_groups(core_proj_bias, 0, core_proj_weight, 0)
    merge_groups(core_proj_weight, 0, qkv_proj_weight, 1)
    core_proj_weight.related_groups[1].typeo = "factor"
    core_proj_weight.related_groups[1].partner = qkv_proj_weight.related_groups[0]
    core_proj_weight.related_groups[1].factor = 1/3
    RelatedGroup.append_to_groups(core_proj_weight, "galerkin attention")
    merge_groups(inp[0], 1, out, 1)
    RelatedGroup.append_to_groups(out, "galerkin attention")

def transpose(inp, dim0, dim1):
    out = torch.transpose(inp, dim0, dim1)

    if hasattr(inp, "related_groups"):
        out.related_groups = list(inp.related_groups)
        out.related_groups[dim0], out.related_groups[dim1] = (
            out.related_groups[dim1],
            out.related_groups[dim0],
        )

    RelatedGroup.append_to_groups(out, "transpose")

    return out


def permute(inp, dims):
    out = torch.permute(inp, dims)

    if hasattr(inp, "related_groups"):
        out.related_groups = [None] * inp.ndim

        for i in range(len(dims)):
            out.related_groups[i] = inp.related_groups[dims[i]]

    RelatedGroup.append_to_groups(out, "permute")

    return out
def mypermute(inp, *dims):
    out = torch.permute(inp, dims)

    if hasattr(inp, "related_groups"):
        out.related_groups = [None] * inp.ndim

        for i in range(len(dims)):
            out.related_groups[i] = inp.related_groups[dims[i]]

    RelatedGroup.append_to_groups(out, "permute")

    return out

def chunk(inp, chunks, dim):
    out = torch.chunk(inp, chunks, dim)

    if hasattr(inp, "related_groups"):
        inp.related_groups[dim].base = chunks
        idm = RelatedGroup(inp.relared_groups[dim].size//3,typeo="factor",partner=inp.related_groups[dim], factor=chunks)
        for i in out:
            i.related_groups = [None] * inp.ndim

            for j in range(inp.ndim):
                i.related_groups[j] = inp.related_groups[j]
            i.related_groups[dim] = idm
            RelatedGroup.append_to_groups(i, "chunk")  
    return out

def getitem(inp, slices):
    out = operator.getitem(inp, slices)

    if hasattr(inp, "related_groups"):
        out.related_groups = [None] * out.ndim
        j = 0

        for i in range(inp.ndim):
            if i < len(slices):  # ADD Ellipsis
                if slices[i] == slice(None):
                    out.related_groups[j] = inp.related_groups[i]
                    j += 1

    RelatedGroup.append_to_groups(out, "getitem")

    return out


def neutral_decorator(call_func):
    def wrapper(*args, **kwargs):
        out = call_func(*args, **kwargs)

        if hasattr(args[0], "related_groups"):
            out.related_groups = args[0].related_groups
            RelatedGroup.append_to_groups(out, "neutral")

        return out

    return wrapper


def conv_linear_decorator(function):
    def conv_linear(*args):
        x, weight, bias = args[:3]
        out = function(*args)

        if bias is not None:
            merge_groups(bias, 0, weight, 0)

        merge_groups(weight, 1, x, 1)
        merge_groups(out, 1, weight, 0)
        RelatedGroup.append_to_groups(out, "conv_linear")

        return out

    return conv_linear


def conv_transposed_decorator(function):
    def conv_transposed(*args):
        x, weight, bias = args[:3]
        out = function(*args)

        if bias is not None:
            merge_groups(bias, 0, weight, 1)

        merge_groups(weight, 0, x, 1)
        merge_groups(out, 1, weight, 1)
        RelatedGroup.append_to_groups(out, "conv_transposed")

        return out

    return conv_transposed


def batch_norm(*args, **kwargs):
    out = torch.nn.functional.batch_norm(*args, **kwargs)
    inp = args[0]
    weight = kwargs["weight"]
    bias = kwargs["bias"]
    merge_groups(inp, 1, weight, 0)
    merge_groups(bias, 0, weight, 0)
    merge_groups(out, 1, weight, 0)
    RelatedGroup.append_to_groups(out, "batch_norm")

    return out


def aggregation_decorator(func):
    def wrapper(inp, *dims, **kwargs):
        out = func(inp, *dims, **kwargs)

        for d in range(out.ndim):
            if d not in dims:
                merge_groups(out, d, inp, d)

        RelatedGroup.append_to_groups(out, "aggregation")

        return out

    return wrapper


def max_min_decorator(func):
    def wrapper(inp, dim, **kwargs):
        out = func(inp, dim, **kwargs)
        values = out.values

        for d in range(values.ndim):
            if d != dim:
                merge_groups(values, d, inp, d)

        RelatedGroup.append_to_groups(values, "min_max")

        return out

    return wrapper


def view(*args, **kwargs):
    inp = args[0]
    out = inp.view(*args[1:])
    out.related_groups = [None] * out.ndim

    if hasattr(inp, "related_groups"):
        i = 1

        for g in inp.related_groups:
            if g is not None:
                while out.shape[i] != g.size:
                    i += 1

                out.related_groups[i] = g
                i += 1

        RelatedGroup.append_to_groups(out)

    return out


def reshape(inp, *shapes):
    out = inp.reshape(*shapes)
    out.related_groups = [None] * out.ndim

    if hasattr(inp, "related_groups") and shapes[1]>inp.shape[1]:
        out.related_groups[-2] = RelatedGroup(0,partner=inp.related_groups[-1],typeo="max")
        out.related_groups[-1] = RelatedGroup(0,partner=inp.related_groups[-1],typeo="min")
        RelatedGroup.append_to_groups(out)

    return out

def layer_norm(inp,normalized_shape,weight,bias,eps):
    out = torch.nn.functional.layer_norm(inp,normalized_shape,weight,bias,eps)
    merge_groups(bias, 0, weight, 0)
    merge_groups(bias, 1, weight, 1)
    merge_groups(inp, 2, weight, 0)
    merge_groups(inp, 3, weight, 1)
    merge_groups(out, 2, weight, 0)
    merge_groups(out, 3, weight, 1)
    RelatedGroup.append_to_groups(out, "layer_norm")
    return out

def concatenate(inputs, dim):
    out = torch.cat(inputs, dim)
    out.related_groups = [None] * out.ndim

    for d in range(out.ndim):
        if d != dim:
            for x in inputs[1:]:
                merge_groups(inputs[0], d, x, d)

            out.related_groups[d] = inputs[0].related_groups[d]

        else:
            out.related_groups[d] = RelatedGroup(out.shape[d])
            out.related_groups[d].set_subgroups([x.related_groups[d] for x in inputs])
    RelatedGroup.append_to_groups(out, "concat")

    return out
def sub_operator(x,y):
    out = operator.sub(x,y)
    out.related_groups = [None] * out.ndim
    for i in range(out.ndim):
        out.related_groups[i] = x.related_groups[i]
    RelatedGroup.append_to_groups(out, "sub")
    return out

def truediv_operator(x,y):
    out = operator.truediv(x, y)
    out.related_groups = [None] * out.ndim
    for i in range(out.ndim):
        out.related_groups[i] = x.related_groups[i]
    RelatedGroup.append_to_groups(out, "truediv")
    return out

def operators_decorator(operator):
    def wrapper(x, y):
        out = operator(x, y)

        if type(x) not in (int, float, torch.Tensor):
            return out

        if type(x) in (int, float):
            x = torch.tensor(x)

        if type(y) in (int, float):
            y = torch.tensor(y)

        if y.ndim > x.ndim:
            x, y = y, x

        k = x.ndim - y.ndim

        for dim in range(y.ndim):
            if x.shape[k + dim] != 1 and y.shape[dim] != 1:
                merge_groups(x, k + dim, y, dim)

        out.related_groups = x.related_groups

        for dim in range(out.ndim):
            if out.related_groups[dim] is None:
                if dim - k >= 0 and y.shape[dim - k] > 1:
                    out.related_groups[dim] = y.related_groups[dim - k]

            if out.shape[dim] == 1:
                out.related_groups[dim] = None

        RelatedGroup.append_to_groups(out, "operator")

        return out

    return wrapper


def matmul(x, y):
    out = x @ y
    out.related_groups = [None] * out.ndim

    if y.ndim > x.ndim:
        y, x = x, y

    k = x.ndim - y.ndim
    merge_groups(y, y.ndim - 2, x, x.ndim - 1)

    for i in range(y.ndim - 2):
        merge_groups(x, i + k, y, i)

    for d in range(x.ndim - 1):
        out.related_groups.append(x.related_groups[d])

    out.related_groups.append(y.related_groups[y.ndim - 1])
    RelatedGroup.append_to_groups(out, "matmul")

    return out


def interpolate(*args, **kwargs):
    out = torch.nn.functional.interpolate(*args, **kwargs)
    out.related_groups = [None] * out.ndim

    if hasattr(args[0], "related_groups"):
        for d in range(out.ndim):
            out.related_groups[d] = args[0].related_groups[d]

    RelatedGroup.append_to_groups(out, "interpolate")

    return out