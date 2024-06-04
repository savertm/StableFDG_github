import random
from contextlib import contextmanager
import torch
import torch.nn as nn
import pdb
from sklearn.cluster import KMeans
import numpy as np

def deactivate_dsu(m):
    if type(m) == DSU:
        m.set_activation_status(False)


def activate_dsu(m):
    if type(m) == DSU:
        m.set_activation_status(True)


def random_dsu(m):
    if type(m) == DSU:
        m.update_mix_method("random")


def crossdomain_dsu(m):
    if type(m) == DSU:
        m.update_mix_method("crossdomain")


@contextmanager
def run_without_dsu(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_dsu)
        yield
    finally:
        model.apply(activate_dsu)


@contextmanager
def run_with_dsu(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == "random":
        model.apply(random_dsu)

    elif mix == "crossdomain":
        model.apply(crossdomain_dsu)

    try:
        model.apply(activate_dsu)
        yield
    finally:
        model.apply(deactivate_dsu)


class DSU(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    # This is in official code#################################
    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * 1
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    ###########################################################

    def forward(self, x,  p=0.5):
        if not self.training or not self._activated:
            return x

        if random.random() > p:
            return x

        B = x.size(0)
        C = x.size(1)

        ###########################################################################################################
        # 1) My code

        # mu = x.mean(dim=[2, 3], keepdim=True)
        # var = x.var(dim=[2, 3], keepdim=True)
        # sig = (var + self.eps).sqrt()
        #
        # #
        # # # mu, sig = mu.detach(), sig.detach() # 1) Is it right to detach the statistics?
        # #
        # # ##################################### 1) our method #######################################################
        # # # Clustering
        # # # n_cluster = 3
        # # # mu_copy = mu.cpu().clone().detach().numpy().reshape(B,C)
        # # # sig_copy = sig.cpu().clone().detach().numpy().reshape(B,C)
        # # # kmeans_mu = KMeans(n_clusters=n_cluster, init='k-means++', random_state=0).fit(mu_copy)
        # # # kmeans_sig = KMeans(n_clusters=n_cluster, init='k-means++', random_state=0).fit(sig_copy)
        # # #
        # # # mu_label = kmeans_mu.labels_
        # # # sig_label = kmeans_sig.labels_
        # # #
        # # # for i in range(n_cluster):
        # # #     if i ==0:
        # # #         mu_mean = mu[np.where(mu_label==i)].mean(dim=[0], keepdim=True)
        # # #         sig_mean = sig[np.where(sig_label == i)].mean(dim=[0], keepdim=True)
        # # #     else:
        # # #         mu_mean += mu[np.where(mu_label == i)].mean(dim=[0], keepdim=True)
        # # #         sig_mean += sig[np.where(sig_label == i)].mean(dim=[0], keepdim=True)
        # # #
        # # # mu_mean = mu_mean/n_cluster
        # # # sig_mean = sig_mean/n_cluster
        # # ############################################################################################
        # #
        # #
        # # ##################################### 2) Naive Mean #######################################################
        # mu_mean = mu.mean(dim=[0], keepdim=True)
        # sig_mean = sig.mean(dim=[0], keepdim=True)
        # #
        # # ############################################################################################
        # #
        # #
        # sig_mu = torch.mean((mu - mu_mean)**2, 0, True)
        # sig_sig = torch.mean((sig - sig_mean)**2, 0, True)
        #
        # #eps_mu = torch.randn_like(sig_mu)    # 1) epsilon_c_dim
        # #eps_sig = torch.randn_like(sig_sig)
        #
        # eps_mu = torch.randn_like(mu)  # 2) epsilon_bc_dim
        # eps_sig = torch.randn_like(sig)
        #
        #
        # #eps_mu = torch.randn(1).to(x.device)  # 3) epsilon_one_dim
        # #eps_sig = torch.randn(1).to(x.device)
        #
        #
        #
        # mu_new = mu + eps_mu * torch.sqrt(sig_mu + self.eps)
        # sig_new = sig + eps_sig * torch.sqrt(sig_sig + self.eps)
        #
        # # pdb.set_trace()
        #
        # x_normed = (x-mu) / sig
        # return x_normed * sig_new + mu_new

        ###############################################################################################################




        # 2) Official code

        mean = x.mean(dim=[2,3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        # pdb.set_trace()

        return x



