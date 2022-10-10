import math
import torch
import argparse

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def args_from_dict(d):
    """Creates (nested) argparse.Namespace objects from (nested) dict.

        :param d: a dict
        :type d: dict
        :returns: (nested) argparse.Namespace object
        :rtype: argparse.Namepsace
        """

    args = argparse.Namespace(**d)
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(args, k, args_from_dict(v))
    return args

def update_args(args, param_dict):
    """Update argparse.Namespace object with parameter dict

    param_dict must match the data structure of args. Attributes of args are replaced with param_dict[attrib_name]
    if type is not dict. If type of an attribute of args is dict, it is updated with param_dict[attrib_name].

    :param args: args obtained with parser.parse_args()
    :param param_dict: A dict with parameters
    :type param_dict: dict
    """

    if isinstance(param_dict, dict):
        for k, v in param_dict.items():
            if hasattr(args, k):
                if isinstance(getattr(args, k), dict):
                    getattr(args, k).update(v)
                elif isinstance(getattr(args, k), argparse.Namespace):
                    update_args(getattr(args, k), v)
                else:
                    if isinstance(v, dict):
                        setattr(args, k, args_from_dict(v))
                    else:
                        setattr(args, k, v)
            else:
                if isinstance(v, dict):
                    setattr(args, k, args_from_dict(v))
                else:
                    setattr(args, k, v)