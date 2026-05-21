import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.base.model import BaseModel

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, x):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        x = x.permute(0, 3, 1, 2)  # [b, c, num_nodes, t]

        
        x_input = self.res_conv(x)
        x_input = x_input[:, :, :, self.kernel_size - 1:]
        out = self.conv1(x) + x_input
        out = out * torch.sigmoid(self.conv2(x))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk, spatial_attention=None):
        # x: [b, c_in, time, n_nodes]
        # Lk: [k, n_nodes, n_nodes] or [n_nodes, n_nodes]
        # spatial_attention: [b, n_nodes, n_nodes]
        if len(Lk.shape) == 2:
            Lk = Lk.unsqueeze(0)

        if spatial_attention is None:
            x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        else:
            attention_weighted_supports = Lk.unsqueeze(0) * spatial_attention.unsqueeze(1)
            x_c = torch.einsum("bknm,bitm->bitkn", attention_weighted_supports, x)

        x_gc = torch.einsum("iok,bitkn->botn", self.theta,
                            x_c) + self.b  # [b, c_out, time, n_nodes]
        return torch.relu(x_gc + x)

class Multi_Head_Spatial_Attention_layer(nn.Module):
    '''
    Multi-head spatial attention.
    Each head learns a different (N, N) relationship map.
    They are averaged at the end → still outputs (B, N, N).
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps, fusion_type, num_heads=4):
        super(Multi_Head_Spatial_Attention_layer, self).__init__()
        self.num_heads = num_heads
        self.fusion_type = fusion_type
        self.head_weights = nn.Parameter(torch.ones(num_heads)) # one scalar per head (ex. num_heads = 4 -> scalar (4,))

        # Each head gets its OWN set of W1, W2, W3, bs, Vs parameters.
        # Think of it as num_heads separate students, each with their own worksheet.
        self.W1 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.W2 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.W3 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.bs = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.Vs = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))
            for _ in range(num_heads)
        ])

    def forward(self, x):
        '''
        :param x: (B, N, F, T)
        :return:  (B, N, N)   ← same shape as before, drop-in replacement
        '''
        head_outputs = []

        for h in range(self.num_heads):
            # ---- exact same math as the original Spatial_Attention_layer ----
            # Step 1: (B,N,F,T) x (T,) → (B,N,F)  then x (F,T) → (B,N,T)
            lhs = torch.matmul(torch.matmul(x, self.W1[h]), self.W2[h])

            # Step 2: (F,) x (B,N,F,T) → (B,N,T) → (B,T,N)
            rhs = torch.matmul(self.W3[h], x).transpose(-1, -2)

            # Step 3: (B,N,T) x (B,T,N) → (B,N,N)
            product = torch.matmul(lhs, rhs)

            # Step 4: sigmoid + bias + Vs weighting → (B,N,N)
            S = torch.matmul(self.Vs[h], torch.sigmoid(product + self.bs[h]))

            # Step 5: softmax → attention weights that sum to 1
            S_normalized = F.softmax(S, dim=1)

            head_outputs.append(S_normalized)  # each is (B, N, N)

        # Stack all heads: list of H tensors (B,N,N) → (B, H, N, N)
        # Then average across the H dimension → (B, N, N)
        # This is the "merge" step — combine all heads into one map
        all_heads = torch.stack(head_outputs, dim=1)  # (B, H, N, N)
        weights = F.softmax(self.head_weights, dim=0)
        H = weights.shape[0]

        weights_reshaped = weights.view(1, H, 1, 1)
        weighted_heads = all_heads * weights_reshaped

        if self.fusion_type == 'mean':
            merged = all_heads.mean(dim=1)                # (B, N, N)
        else:
            merged = weighted_heads.sum(dim=1)

        return merged

class STGCN_MHSA_Block(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self,
                 DEVICE,
                 in_channels,
                 num_of_vertices,
                 num_of_timesteps,
                 spatial_channels,
                 out_channels,
                 num_nodes,
                 supports_len,
                 num_heads,
                 fusion_type):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCN_MHSA_Block, self).__init__()
        self.num_heads = num_heads
        self.fusion_type = fusion_type

        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
        #  spatial_channels))

        attention_timesteps = num_of_timesteps - (self.temporal1.kernel_size - 1)
        self.SAt = Multi_Head_Spatial_Attention_layer(
            DEVICE,
            out_channels,
            num_of_vertices,
            attention_timesteps,
            fusion_type,
            num_heads=num_heads,
        )

        self.spatial = SpatialBlock(supports_len, out_channels, spatial_channels)

        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.layer_norm = nn.LayerNorm([num_nodes, out_channels])
        self.layer_norm = nn.LayerNorm([out_channels])
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.Theta1.shape[1])
        # self.Theta1.data.uniform_(-stdv, stdv)
        pass

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # print('to temporal1:')
        # print(X.shape)     #[16, 1085, 12, 27] 

        t = self.temporal1(X)  #[16, 1085, 22, 64]
        
        t_for_attention = t.permute(0, 1, 3, 2)

        spatial_At = self.SAt(t_for_attention)

        # print('to spatial:')
        # print(t.shape)

        # # original
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t2 = self.spatial(t.permute(0, 3, 2, 1), A_hat) #[16, 64, 22, 1085]

        t2 = self.spatial(t.permute(0, 3, 2, 1), A_hat, spatial_At)

        # print('to temporal2:')
        # print(t2.shape)

        t3 = self.temporal2(t2.permute(0, 3, 2, 1)) #[16, 1085, 20, 64]

        # print('to layer norm:')
        # print(t3.shape)
        
        # for layer norm
        t3 = t3.permute(0, 2, 1, 3)
        out = self.layer_norm(t3) #[16, 20, 1085, 64]
        # print(out.shape)
        # input()
        return out.permute(0, 2, 1, 3)
        # return t3


class STGCN_MHSA(BaseModel):
    def __init__(self, n_filters, supports_len, num_heads=2, fusion_type='mean', **args):
        super(STGCN_MHSA, self).__init__(**args)
        self.n_filters = n_filters
        self.supports_len = supports_len
        self.block1 = STGCN_MHSA_Block(
            DEVICE=self.device,
            in_channels=self.input_dim,
            num_of_vertices=self.num_nodes,
            num_of_timesteps=self.seq_len,
            spatial_channels=self.n_filters,
            out_channels=self.n_filters,
            num_nodes=self.num_nodes,
            supports_len=self.supports_len,
            num_heads=num_heads,
            fusion_type=fusion_type,
        )
        self.block2 = STGCN_MHSA_Block(
            DEVICE=self.device,
            in_channels=self.n_filters,
            num_of_vertices=self.num_nodes,
            num_of_timesteps=self.seq_len - 4,
            spatial_channels=self.n_filters,
            out_channels=self.n_filters,
            num_nodes=self.num_nodes,
            supports_len=self.supports_len,
            num_heads=num_heads,
            fusion_type=fusion_type,
        )
        self.last_temporal = TimeBlock(
            in_channels=self.n_filters, out_channels=self.n_filters)
        self.fully = nn.Linear((self.seq_len - 2 * 5) * self.n_filters,
                               self.horizon*self.output_dim)


    def forward(self, X, supports):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """

        # print(X.shape) #[b, l, n, d]

        X = X.permute(0, 2, 1, 3)
        out1 = self.block1(X, supports) #[16, 1085, 20, 64]
        out2 = self.block2(out1, supports) # [16, 1085, 16, 64]
        out3 = self.last_temporal(out2) # [16, 1085, 14, 64]
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1))) #[16, 1085, 24]
        
        # print("out1: {}, out2: {}, out3: {}, out4: {}".format(out1.shape, out2.shape, out3.shape, out4.shape))
        # input()
        b, n, tc = out4.shape
        return out4.reshape(b, n, -1, self.output_dim).permute(0, 2, 1, 3)

        # print('out shape:')
        # print(out4.shape)
        # return out4
