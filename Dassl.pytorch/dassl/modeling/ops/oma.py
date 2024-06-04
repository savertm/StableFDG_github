import random
from contextlib import contextmanager
import torch
import torch.nn as nn
import numpy as np

def deactivate_oma(m):
    if type(m) == OMA:
        m.set_activation_status(False)

def activate_oma(m):
    if type(m) == OMA:
        m.set_activation_status(True)


def random_oma(m):
    if type(m) == OMA:
        m.update_mix_method("random")


def crossdomain_oma(m):
    if type(m) == OMA:
        m.update_mix_method("crossdomain")


@contextmanager
def run_without_oma(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_oma)
        yield
    finally:
        model.apply(activate_oma)


@contextmanager
def run_with_oma(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == "random":
        model.apply(random_oma)

    elif mix == "crossdomain":
        model.apply(crossdomain_oma)

    try:
        model.apply(activate_oma)
        yield
    finally:
        model.apply(deactivate_oma)

#Style Exploration
class OMA(nn.Module):

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):

        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.beta_mixup = torch.distributions.Beta(0.1, 0.1)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.indicator = 0
        self.when = 0

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def to_one_hot(self, inp, num_classes):
        y_onehot = torch.FloatTensor(inp.size(0), num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
        return y_onehot.to(inp.device)

    def forward(self, x, domain, label, supplemental_samples=None, param=None, layer_mix=1):

        if not self.training or not self._activated:
            return x, label

        if random.random() > 0.5:
            return x, label

        self.indicator += 1

        exploration_level = param[0]
        oversampling_size = param[1]

        if self.indicator == 1:
            if supplemental_samples is not None:
                feat_sup = supplemental_samples[0][layer_mix-1]
                label_sup = supplemental_samples[1]

            B_ori = x.size(0)

            unique_label = torch.unique(label)
            unique_label = unique_label.cpu().numpy()
            ind_per_label = {}
            num_per_label_dic = {}
            for i in unique_label:
                ind_tmp = (label==i).nonzero().view(-1).cpu().numpy()
                ind_per_label[i] = ind_tmp
                num_per_label_dic[i] = ind_tmp.shape[0]

            class_sup = [k for k, v in num_per_label_dic.items() if v == 1]

            # Only for Office-Home dataset
            if len(class_sup) != 0:
                if len(class_sup) <= oversampling_size:
                    supplemental_index = [(label_sup==i).nonzero()[0].item() for i in class_sup]
                    x = torch.vstack([x, feat_sup[supplemental_index]])
                    label = torch.hstack([label, label_sup[supplemental_index]])

                else:
                    supplemental_index = [(label_sup==i).nonzero()[0].item() for i in class_sup]
                    sample_ind = np.random.choice(supplemental_index, oversampling_size, replace=False).tolist()
                    x = torch.vstack([x, feat_sup[sample_ind]])
                    label = torch.hstack([label, label_sup[sample_ind]])

                for i in unique_label:
                    ind_tmp = (label == i).nonzero().view(-1).cpu().numpy()
                    ind_per_label[i] = ind_tmp
                    num_per_label_dic[i] = ind_tmp.shape[0]

            num_per_label = np.array([i for i in num_per_label_dic.values()])
            batch_limit = oversampling_size - len(class_sup)
            index_class = []
            add_index = []

            if batch_limit > 0:

                while True:
                    unique_num_samples = np.unique(num_per_label)
                    k = unique_num_samples[0]
                    candi = (num_per_label == k).nonzero()[0]
                    if candi.shape[0] >= batch_limit:
                        index_a = np.random.choice(candi, batch_limit, replace=False).tolist()
                        index_class += list(unique_label[index_a])
                        break
                    else:
                        index_a = np.random.choice(candi, candi.shape[0], replace=False).tolist()
                        index_class += list(unique_label[index_a])
                        num_per_label += (num_per_label == k)
                        batch_limit -= candi.shape[0]

                for i in index_class:
                    add_index  += np.random.choice(ind_per_label[i], 1, replace=False).tolist()

            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()

            x_normed = (x - mu) / sig
            x_normed[B_ori:] = x_normed[B_ori:]
            x_norm_oversampled = x_normed[add_index]

            x_normed = torch.vstack([x_normed, x_norm_oversampled])
            label = torch.hstack([label, label[add_index]])

            # Style Exploration
            mu_mean = mu.mean(dim=[0], keepdim=True)
            sig_mean = sig.mean(dim=[0], keepdim=True)

            mu_mix = (mu[add_index] - mu_mean) * exploration_level + mu[add_index]
            sig_mix = (sig[add_index] - sig_mean) * exploration_level + sig[add_index]
            # sig_mix = (sig_mix >= 0) * sig_mix

            mu_extended = torch.vstack([mu,  mu_mix])
            sig_extended = torch.vstack([sig,  sig_mix])

        else:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()

            x_normed = (x - mu) / sig
            mu_extended = mu
            sig_extended = sig

        B = x_normed.size(0)

        # Mixing the Styles
        mu_extended, sig_extended = mu_extended.detach(), sig_extended.detach()
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)
        mu2, sig2 = mu_extended[perm], sig_extended[perm]
        mu_mix = mu_extended * lmda + mu2 * (1 - lmda)
        sig_mix = sig_extended * lmda + sig2 * (1 - lmda)

        return x_normed*sig_mix + mu_mix, label


