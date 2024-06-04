import torch
import random
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

#Style Sharing
def style_insert(x, style=None, num_blocks=None):

    if style is None:
        return x

    else:
        if random.random() > 0.5:
            return x

        else:
            share_number = 16 # half of the batch
            B = x.size(0)
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + 1e-6).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            if num_blocks == 8: # ResNet-18
                chan = 128
            else: # ResNet-50
                chan = 512

            stat = torch.cat((mu, sig), dim=1)
            stat = stat.view(B, chan)
            stat = kmeans_plus(stat, x.device, share_number)

            mu_sel = stat[:,:int(chan/2)].view(x.shape[0] - share_number, int(chan/2), 1, 1)
            sig_sel = stat[:,int(chan/2):].view(x.shape[0] - share_number, int(chan/2), 1, 1)
            style1 = style[0].to(x.device)

            try:
                style2 = style[1].to(x.device)

            except:
                style2 = 0

            mu_mean_other = style1[:int(chan/2)]
            sig_mean_other = style1[int(chan/2):]

            if torch.sum(style2) == 0:
                new_mu = mu_mean_other.repeat(share_number, 1)  # B,C,1,1
                new_sig = sig_mean_other.repeat(share_number, 1)

            else:
                mu_sig_other = style2[:int(chan/2)]
                sig_sig_other = style2[int(chan/2):]

                mu_sig_other = mu_sig_other.view(1, int(chan/2))
                sig_sig_other = sig_sig_other.view(1, int(chan/2))

                mu_sig_other = mu_sig_other.repeat(share_number, 1)  # B,C,1,1
                sig_sig_other = sig_sig_other.repeat(share_number, 1)

                new_mu = torch.randn_like(mu_sig_other) * mu_sig_other + mu_mean_other
                new_sig = torch.randn_like(sig_sig_other) * sig_sig_other + sig_mean_other

            new_sig = new_sig * (new_sig>=0)

            mu_new = torch.vstack((mu_sel, new_mu.view(share_number, int(chan/2), 1, 1)))
            sig_new = torch.vstack((sig_sel, new_sig.view(share_number, int(chan/2), 1, 1)))

            return x_normed * sig_new + mu_new


def kmeans_plus(x, device, share_number):
    x = x.data.cpu().numpy()

    model = KMeans(n_clusters=x.shape[0] - share_number, init='k-means++', max_iter=1, n_init=1)
    model.fit(x)

    center = model.cluster_centers_
    stat = torch.tensor(center).to(device)

    return stat
