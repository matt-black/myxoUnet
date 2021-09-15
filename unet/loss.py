"""Loss functions
"""
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from util import one_hot


class DiceLoss(nn.Module):
    
    def __init__(self, reduction="mean", eps=1e-12):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        self.reduction = reduction
        
    def forward(self, pred, target):
        return dice_loss(pred, target, self.reduction, self.eps)


def dice_loss(pred, target, reduction="mean", eps=1e-12):
    """
    Criterion using the Sorensen-Dice coefficient

    Parameters
    ----------
    pred : torch.Tensor
        (B, C, H, W) probability map of class predictions
    target : torch.Tensor
        (B, H, W) ground-truth class label tensor
        dtype should be torch.int64 (Long)
        
    Returns
    -------
    Dice loss between predictions and target
    """
    _check_valid_reduction(reduction)
    # convert target to one hot
    target_1h = one_hot(target, pred.size(1), 
                        device=pred.device, dtype=pred.dtype)
    # compute intersection/cardinality
    inter = torch.sum(pred * target_1h, (1, 2, 3))
    card  = torch.sum(pred + target_1h, (1, 2, 3))
    dice  = 2 * inter / (card + eps)
    # choose reduction
    if reduction == "mean":
        return torch.mean(1.0 - dice)
    elif reduction == "sum":
        return torch.sum(1.0 - dice)
    else:
        return (1.0 - dice)
    
class DiceRegularizedCrossEntropy(nn.Module):
    
    def __init__(self, reduction="mean", eps=1e-6):
        super(DiceRegularizedCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, pred, target):
        return dice_regularized_cross_entropy(pred, target,
                                              reduction=self.reduction,
                                              eps=self.eps)
    
    
def dice_regularized_cross_entropy(pred, target, reduction="mean", eps=1e-6):
    dice = dice_loss(pred, target, reduction="none", eps=eps)
    ce = F.cross_entropy(_collapse_outer_dims(pred), 
                         _collapse_outer_dims(target), 
                         reduction="none")
    ce = ce.mean(dim=-1)
    if reduction == "mean":
        return (dice + ce).mean()
    elif reduction == "sum":
        return (dice + ce).sum()
    else:
        return dice + ce
    

class JLoss(nn.Module):
    
    def __init__(self, weight=None, reduction="none"):
        super(JLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        return j_loss(pred, target, 
                      weight=self.weight, 
                      reduction=self.reduction)


def j_loss(pred, target, weights=None, reduction="none"):
    """
    J-statistic based loss function

    from, arXiv:1910.09783v1

    Parameters
    ----------
    pred : torch.Tensor
        (B, C, H, W) probability map of class predictions
    target : torch.Tensor
        (B, H, W) ground-truth class label tensor
        dtype should be torch.int64 (Long)
        

    Returns
    -------
    J loss for prediction & targets
    """
    if weights is None:
        weights = torch.ones((pred.size(1), pred.size(1)), 
                             device=pred.device, dtype=pred.dtype)
    
    # convert target to one hot
    num_class = pred.size(1)
    target_1h = one_hot(target, num_class, 
                        device=pred.device, dtype=pred.dtype)
    
    # compute phi
    n_i = target_1h.sum(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
    phi = torch.div(target_1h, n_i)
    
    # compute per-class loss
    loss = torch.zeros((pred.size(0), num_class, num_class),
                       device=pred.device, dtype=pred.dtype)
    for ci in range(pred.size(1)):
        for ck in range(pred.size(1)):
            if ci == ck:
                continue
            delta_ik = torch.div(phi[:,ci,:,:]-phi[:,ck,:,:], 2)
            loss[:,ci,ck] = weights[ci,ck] * torch.log(
                0.5 + torch.mul(pred[:,ci,:,:], delta_ik).sum(dim=(-2,-1)))
    # sum over classes
    j = torch.neg(loss.sum(dim=(-2, -1)))
    
    if reduction == "mean":
        return j.mean()
    elif reduction == "sum":
        return j.sum()
    else:
        return j
    
    
class JRegularizedCrossEntropyLoss(nn.Module):
    
    def __init__(self, j_wgt=None, ce_wgt=None, reduction="mean"):
        super(JRegularizedCrossEntropyLoss, self).__init__()
        self.j_wgt = j_wgt
        self.ce_wgt = ce_wgt
        self.reduction = reduction

    def forward(self, pred, target):
        return j_regularized_cross_entropy(pred, target, 
                                           j_wgt=self.j_wgt, 
                                           ce_wgt=self.ce_wgt,
                                           reduction=self.reduction)
    
    
def j_regularized_cross_entropy(pred, target, j_wgt=None, ce_wgt=None,
                                reduction="none"):
    """
    """
    _check_valid_reduction(reduction)
    jl = j_loss(pred, target, j_wgt, reduction)
    if ce_wgt is None:
        ce = F.cross_entropy(_collapse_outer_dims(pred), 
                             _collapse_outer_dims(target), 
                             reduction=reduction)
    else:
        ce = F.cross_entropy(_collapse_outer_dims(pred),
                             _collapse_outer_dims(target),
                             weight=ce_wgt, reduction=reduction)    
    return jl + ce


class SSIM(nn.Module):
    r"""Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    Examples::

        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = tgm.losses.SSIM(5, reduction='none')
        >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(self,
                 window_size: int,
                 reduction: str = 'none',
                 max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.padding: int = self.compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    @staticmethod
    def compute_zero_padding(kernel_size: int) -> int:
        """Computes zero padding."""
        return (kernel_size - 1) // 2

    def filter2D(self,
                 input: torch.Tensor,
                 kernel: torch.Tensor,
                 channel: int) -> torch.Tensor:
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self,
                img1: torch.Tensor,
                img2: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                            .format(type(img1)))
        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                            .format(type(img2)))
        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                             .format(img1.shape))
        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                             .format(img2.shape))
        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}"
                             .format(img1.shape, img2.shape))
        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}"
                             .format(img1.device, img2.device))
        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}"
                             .format(img1.dtype, img2.dtype))
        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # compute local mean per channel
        mu1: torch.Tensor = self.filter2D(img1, kernel, c)
        mu2: torch.Tensor = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss


def ssim(img1: torch.Tensor,
         img2: torch.Tensor,
         window_size: int,
         reduction: str = 'none',
         max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`SSIM` for details.
    """
    return SSIM(window_size, reduction, max_val)(img1, img2)


class DSSIMLoss(nn.Module):
    def __init__(self,
                 window_size: int,
                 reduction: str = 'none',
                 max_val: float = 1.0) -> None:
        super(DSSIMLoss, self).__init__()
        self.ssim = SSIM(window_size, reduction, max_val)

    def forward(self, pred, target):
        return (1 - self.ssim(pred, target)) / 2


def dssim_loss(pred, target, window_size,
               reduction='none', max_val=1.0):
    return (1- ssim(pred, target, window_size, reduction, maxval)) / 2


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x)))
         for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        ksize (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.
    """
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d


def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        ksize (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(ksize_x, ksize_y)`
    """
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def _collapse_outer_dims(x):
    """
    collapse all dims past the first two into a single one

    Parameters
    ----------
    x : torch.Tensor
        (BxCx...)
    Returns
    -------
    `x` reshaped to (BxCxN)

    """
    assert len(x.shape) == 3 or len(x.shape) == 4
    if len(x.shape) == 4:
        new_shape = (x.shape[0], x.shape[1],
                     torch.mul(*x.shape[2:]))
    else:
         new_shape = (x.shape[0], torch.mul(*x.shape[1:]))   
    return torch.reshape(x, new_shape)


def _check_valid_reduction(reduction):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            "invalid reduction, {}. Valid are 'mean', 'sum', 'none'".format(
                reduction))    
