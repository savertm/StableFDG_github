# This code is built upon the "Attention-Augmented-Conv2d" code
# URL: https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/in_paper_attention_augmented_conv/attention_augmented_conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention-based Feature Highlighter
class AFH(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, relative):
        super(AFH, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = nn.Conv2d(1, 1, 1)

    def forward(self, x, domain, label, supplemental_samples=None, tmp_=False, layer=2):

        batch, _, height, width = x.size()

        if self.training:
            if supplemental_samples is not None:
                flat_q_m, flat_k_m, flat_v_m = supplemental_samples[2][0]

            unique_label = torch.unique(label)
            unique_label = unique_label.cpu().numpy()
            ind_per_label = {}
            num_per_label_dic = {}
            for i in unique_label:
                ind_tmp = (label==i).nonzero().view(-1).cpu().numpy()
                ind_per_label[i] = ind_tmp
                num_per_label_dic[i] = ind_tmp.shape[0]

            pair_index = []
            pair_index_1s = []
            pair_index_1s_class = []

            for j in range(batch):

                if num_per_label_dic[label[j].item()] == 1:
                    add_ind = ind_per_label[label[j].item()][0]
                    pair_index_1s.append(add_ind)
                    pair_index_1s_class.append(label[j].item())

                else:
                    tmp = (ind_per_label[label[j].item()]==j).nonzero()[0]
                    add_ind = ind_per_label[label[j].item()][(tmp+1) % num_per_label_dic[label[j].item()]][0]

                pair_index.append(add_ind)
        else:
            pair_index = [k for k in range(batch)]

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        if tmp_ is True:
            return x, [flat_q, flat_k, flat_v]

        if self.training:
            if len(pair_index_1s) != 0:
                flat_q[pair_index_1s] = flat_q_m[pair_index_1s_class]

        flat_q = flat_q / flat_q.norm(dim=2)[:,:,None,:]
        flat_k = flat_k / flat_k.norm(dim=2)[:,:,None,:]

        logits = torch.matmul(((flat_q[pair_index] + flat_q)/2).transpose(2, 3), flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        attn_out = torch.reshape(logits, (batch, self.Nh, height * width, height, width))
        attn_out = self.combine_heads_2d(attn_out)  # (batch, out_channels, height, width)

        attn_out = torch.mean(attn_out, dim=[1], keepdim=True)

        out = F.softmax((attn_out).reshape(attn_out.size(0), attn_out.size(1), -1), 2).view_as(attn_out)

        return torch.cat((x/49, x * out), dim=1)


    def compute_flat_qkv(self, x, dk, dv, Nh):
        N, _, H, W = x.size()
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W)) # flatten HW
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width) # split head
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        key_rel_w = nn.Parameter(torch.randn((2 * W - 1, dk), requires_grad=True)).to(q.device)
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, "w")

        key_rel_h = nn.Parameter(torch.randn((2 * H - 1, dk), requires_grad=True)).to(q.device)
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x.device)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

