import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from Libs.utils import get_layer
from torch.nn.parameter import Parameter


class Generator(nn.Module):
    def __init__(self,
                 n_fft=320,
                 n_band=1,
                 n_layers=4,
                 lstm_hidden_units=256,
                 attn_n_head=4,
                 attn_approx_qk_dim=512,
                 emb_dim=32,
                 emb_ks=8,
                 emb_hs=8,
                 activation="prelu",
                 eps=1.0e-5):
        super().__init__()
        self.n_layers = n_layers
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 // n_band + 1

        # Speech Encoder
        t_ksize, f_ksize = 3, 3
        ks, padding = (t_ksize, f_ksize), (t_ksize // 2, f_ksize // 2)
        self.conv = nn.Sequential(nn.Conv2d(n_band * 2, emb_dim, ks, padding=padding),
                                  nn.GroupNorm(1, emb_dim, eps=eps))

        self.restoration_blocks = nn.Sequential(*[CausalGridNetBlock(emb_dim,
                                                                        emb_ks,
                                                                        emb_hs,
                                                                        n_freqs,
                                                                        lstm_hidden_units,
                                                                        n_head=attn_n_head,
                                                                        approx_qk_dim=attn_approx_qk_dim,
                                                                        activation=activation,
                                                                        eps=eps) for _ in range(n_layers)])

        # Speech Decoder
        self.deconv = nn.ConvTranspose2d(emb_dim, n_band * 2, ks, padding=padding)


    def forward(self,
                x: Tensor) -> Tensor:
        """Forward.
            Args:
                x (Tensor): batched audio tensor with N samples [B, 2, T, F].
            Returns:
                preprocessed (Tensor): [B, 2, T, F] audio tensors with N samples.
        """
        esti = self.conv(x)  # [B, -1, T, F]
        esti = self.restoration_blocks(esti)  # [B, -1, T, F]
        esti = self.deconv(esti)  # [B, 2, T, F]
        
        return esti


class CausalGridNetBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 emb_ks,
                 emb_hs,
                 n_freqs,
                 hidden_channels,
                 n_head=4,
                 approx_qk_dim=512,
                 activation='prelu',
                 eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        self.lookback = 5
        self.lookahead = 0

        # Intra-Frame Full-Band Module
        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(2 * hidden_channels,
                                               emb_dim,
                                               kernel_size=emb_ks,
                                               stride=emb_hs)

        # Sub-Band Temporal Module
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(in_channels,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False)
        self.inter_linear = nn.ConvTranspose1d(hidden_channels,
                                               emb_dim,
                                               kernel_size=emb_ks,
                                               stride=emb_hs)

        # Cross-Frame Self-Attention Module
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0

        for ii in range(n_head):
            self.add_module(f"attn_conv_Q_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_K_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_V_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, emb_dim // n_head, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps)))
        self.add_module("attn_concat_proj",
                        nn.Sequential(nn.Conv2d(emb_dim, emb_dim, kernel_size=1),
                                      get_layer(activation)(),
                                      LayerNormalization4DCF((emb_dim, n_freqs), eps=eps)))


    def __getitem__(self, item):
        return getattr(self, item)

    def forward(self, x):
        """GridNetBlock Forward.
            Args:
                x: [B, C, T, F]
            Returns:
                out: [B, C, T, F]
        """
        B, C, old_T, old_F = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        F = math.ceil((old_F - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = nn.functional.pad(x, (0, F - old_F, 0, T - old_T))

        # Intra-Frame Full-Band Module
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, F]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, F)  # [BT, C, F]
        intra_rnn = nn.functional.unfold(intra_rnn[..., None],
                                         (self.emb_ks, 1),
                                         stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_rnn(intra_rnn)[0]  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, F]
        intra_rnn = intra_rnn.view([B, T, C, F])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, F]
        intra_rnn = intra_rnn + input_  # [B, C, T, F]

        # Sub-Band Temporal Module
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * F, C, T)  # [BF, C, T]
        inter_rnn = nn.functional.unfold(inter_rnn[..., None],
                                         (self.emb_ks, 1),
                                         stride=(self.emb_hs, 1))  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn = self.inter_rnn(inter_rnn)[0]  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, F, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, F]
        inter_rnn = inter_rnn + input_  # [B, C, T, F]

        # Cross-Frame Self-Attention Module
        inter_rnn = inter_rnn[..., :old_T, :old_F]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self[f"attn_conv_Q_{ii}"](batch))  # [B, C, T, F]
            all_K.append(self[f"attn_conv_K_{ii}"](batch))  # [B, C, T, F]
            all_V.append(self[f"attn_conv_V_{ii}"](batch))  # [B, C, T, F/H]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, F]
        K = torch.cat(all_K, dim=0)  # [B', C, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C, T, F/H]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*F]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*F]
        V = V.transpose(1, 2)  # [B', T, C, F/H]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/H]
        emb_dim = Q.shape[-1]

        # causal mask
        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / emb_dim ** 0.5  # [B', T, T]
        attn_mat = self.causal_mask(attn_mat)

        # attn_mat = nn.functional.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/H]

        V = V.reshape(old_shape)  # [B', T, C, F/H]
        V = V.transpose(1, 2)  # [B', C, T, F/H]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [H, B, C, T, F/H]
        batch = batch.transpose(0, 1)  # [B, H, C, T, F/H])
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])  # [B, C, T, F]
        batch = self["attn_concat_proj"](batch)  # [B, C, T, F]
        out = batch + inter_rnn

        return out
    
    def causal_mask(self, attn_mat):
        old_T = attn_mat.shape[-1]
        mask_attn_mat = torch.zeros_like(attn_mat)
        
        for t in range(old_T):
            start = max(0, t - self.lookback)
            end = t + 1 + self.lookahead
            mask_attn_mat[:, t, start: end] = nn.functional.softmax(attn_mat[:, t, start: end], dim=-1)

        return mask_attn_mat


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B,1,T,F]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B,1,T,1]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta
        return x_hat



if __name__ == "__main__":
    import toml

    configs = toml.load('Configs/train_config.toml')
    gpuids = tuple(configs['gpu']['gpu_ids'])
    device = torch.device("cuda:{}".format(gpuids[0]))

    net = Generator(n_fft=configs['signal']['fft_num'],
                    n_band=configs['signal']['band_split'],
                    n_layers=configs['net']['generator']['n_layers'],
                    lstm_hidden_units=configs['net']['generator']['lstm_hidden_units'],
                    attn_n_head=configs['net']['generator']['attn_n_head'],
                    attn_approx_qk_dim=configs['net']['generator']['attn_approx_qk_dim'],
                    emb_dim=configs['net']['generator']['emb_dim'],
                    emb_ks=configs['net']['generator']['emb_ks'],
                    emb_hs=configs['net']['generator']['emb_hs'],
                    activation=configs['net']['generator']['activation'],
                    eps=configs['net']['generator']['eps']).to(device)
    
    # num_params = sum([param.nelement() for param in net.parameters()]) / 10.0 ** 6
    # print(num_params)
    
    from ptflops import get_model_complexity_info

    def input_constructor(input_shape):
        inputs = {'x': torch.ones((1, input_shape[0], input_shape[1], input_shape[2]), device=device)}
        
        return inputs


    macs, params = get_model_complexity_info(net, (2, 401, 161),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             input_constructor=input_constructor,
                                             backend='pytorch',
                                             verbose=True,
                                             output_precision=4)
    print('{:<30} {:<8}'.format('Computational complexity: ', macs))
    print('{:<30} {:<8}'.format('Number of parameters: ', params))
