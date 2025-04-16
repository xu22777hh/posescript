import torch
import torch.nn.functional as F
import numpy as np


def BBC(scores):
    # build the ground truth label tensor: the diagonal corresponds to the
    # correct classification
    GT_labels = torch.arange(scores.shape[0], device=scores.device).long()
    loss = F.cross_entropy(scores, GT_labels) # mean reduction
    return loss


def symBBC(scores):
    x2y_loss = BBC(scores)
    y2x_loss = BBC(scores.t())
    return (x2y_loss + y2x_loss) / 2.0


def laplacian_nll(x_tilde, x, log_sigma):
    """ Negative log likelihood of an isotropic Laplacian density """
    log_norm = - (np.log(2) + log_sigma)
    log_energy = - (torch.abs(x_tilde - x)) / torch.exp(log_sigma)
    return - (log_norm + log_energy)


def gaussian_nll(x_tilde, x, log_sigma):
    """ Negative log-likelihood of an isotropic Gaussian density """
    log_norm = - 0.5 * (np.log(2 * np.pi) + log_sigma)
    log_energy = - 0.5 * F.mse_loss(x_tilde, x, reduction='none') / torch.exp(log_sigma)
    return - (log_norm + log_energy)



class SigLipLoss(torch.nn.Module):
    def __init__(self, logit_scale: float = np.log(10), logit_bias: float = -10):
        super().__init__()
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias

    def get_ground_truth(self, device: torch.device, dtype: torch.dtype, num_logits: int, negative_only: bool = False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: float, logit_bias: float = None) -> torch.Tensor:
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, negative_only: bool = False) -> torch.Tensor:
        logits = self.get_logits(image_features, text_features, self.logit_scale, self.logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss
