"""Utility classes/functions
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def one_hot(labels, num_class, device, dtype, eps=1e-12):
    """
    Convert the input label vector/image to its equivalent one-hot encoding
    Parameters
    ----------
    labels : torch.LongTensor
        input label image/vector to convert to one-hot
    num_class : int
        number of classes
    device : torch.device
        DESCRIPTION.
    dtype : torch.dtype
        output datatype
    eps : float, optional
        epsilon value to prevent weirdness. The default is 1e-6.

    Raises
    ------
    ValueError
        invalid label shape.
    TypeError
        if input is not LongTensor.

    Returns
    -------
    torch.Tensor
        one-hot encoding of input label image/vector.

    """
    # input integrity
    if not len(labels.shape) == 3:
        raise ValueError(
            "invalid label shape, should be BxHxW. Got {}".format(
            labels.size()))
    if not labels.dtype == torch.int64:
        raise TypeError("labels must be type torch.int64 (Long)")
    batch_size, num_row, num_col = labels.size()
    one_hot = torch.zeros(batch_size, num_class, num_row, num_col,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def convTranspose2dOutputSize(dim, kernel_size, stride=1, padding=0, 
                              dilation=1, output_padding=0):
    return (dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) \
        + output_padding + 1
        

def conv2dOutputSize(dim, kernel_size, stride=1, padding=0, dilation=1):
    return ((dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) \
        + 1
        

class AvgValueTracker(object):
    """Computes/stores average & current value
    
    NOTE: ripped from PyTorch imagenet example
    """
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
        
    
class ProgressShower(object):
    """Show progress of training
    
    NOTE: ripped from PyTorch imagenet example
    """
    def __init__(self, n_batch, trackers, prefix=""):
        self.fmt = self._get_batch_fmtstr(n_batch)
        self.trackers = trackers
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.fmt.format(batch)]
        entries += [str(t) for t in self.trackers]
        print("\t".join(entries))
    
    def _get_batch_fmtstr(self, n_batch):
        n_dig = len(str(n_batch // 1))
        fmt = "{:" + str(n_dig) + "d}"
        return "[" + fmt + "/" + fmt.format(n_batch) + "]"
