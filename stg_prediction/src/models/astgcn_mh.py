# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel
# from src.layers.embedding import AirEmbedding


'''

Approach 1: Multi-head spatial attention

Purpose:
let different heads learn different kinds of region-to-region relationships

Why it helps:
one head may capture nearby zones
another may capture same-demand-pattern zones
another may capture CBD -> suburb spillover
another may capture simultaneous peaks across business areas

Why this is a nice upgrade:
the current Spatial_Attention_layer gives one single (N, N) attention map
multi-head attention gives several parallel maps, which is much more expressive

Paper angle:
different heads capture heterogeneous urban demand interactions

'''

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(
            in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(
            1, num_of_vertices, num_of_vertices).to(DEVICE)) # many parameters
        self.Vs = nn.Parameter(torch.FloatTensor(
            num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)

        # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(
            product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized

# class Multi_Head_Spatial_Attention_layer(nn.Module):
#     '''
#     Multi-head spatial attention.
#     Each head learns a different (N, N) relationship map.
#     They are averaged at the end → still outputs (B, N, N).
#     '''
#     def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps, num_heads=4):
#         super(Multi_Head_Spatial_Attention_layer, self).__init__()
#         self.num_heads = num_heads

#         # Each head gets its OWN set of W1, W2, W3, bs, Vs parameters.
#         # Think of it as num_heads separate students, each with their own worksheet.
#         self.W1 = nn.ParameterList([
#             nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
#             for _ in range(num_heads)
#         ])
#         self.W2 = nn.ParameterList([
#             nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
#             for _ in range(num_heads)
#         ])
#         self.W3 = nn.ParameterList([
#             nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
#             for _ in range(num_heads)
#         ])
#         self.bs = nn.ParameterList([
#             nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
#             for _ in range(num_heads)
#         ])
#         self.Vs = nn.ParameterList([
#             nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))
#             for _ in range(num_heads)
#         ])

#     def forward(self, x):
#         '''
#         :param x: (B, N, F, T)
#         :return:  (B, N, N)   ← same shape as before, drop-in replacement
#         '''
#         head_outputs = []

#         for h in range(self.num_heads):
#             # ---- exact same math as the original Spatial_Attention_layer ----
#             # Step 1: (B,N,F,T) x (T,) → (B,N,F)  then x (F,T) → (B,N,T)
#             lhs = torch.matmul(torch.matmul(x, self.W1[h]), self.W2[h])

#             # Step 2: (F,) x (B,N,F,T) → (B,N,T) → (B,T,N)
#             rhs = torch.matmul(self.W3[h], x).transpose(-1, -2)

#             # Step 3: (B,N,T) x (B,T,N) → (B,N,N)
#             product = torch.matmul(lhs, rhs)

#             # Step 4: sigmoid + bias + Vs weighting → (B,N,N)
#             S = torch.matmul(self.Vs[h], torch.sigmoid(product + self.bs[h]))

#             # Step 5: softmax → attention weights that sum to 1
#             S_normalized = F.softmax(S, dim=1)

#             head_outputs.append(S_normalized)  # each is (B, N, N)

#         # Stack all heads: list of H tensors (B,N,N) → (B, H, N, N)
#         # Then average across the H dimension → (B, N, N)
#         # This is the "merge" step — combine all heads into one map
#         all_heads = torch.stack(head_outputs, dim=1)  # (B, H, N, N)
#         merged = all_heads.mean(dim=1)                # (B, N, N)

#         return merged


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(
            in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                T_k_with_at = T_k.mul(spatial_attention)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)

                # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
    

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


# class Temporal_Attention_layer(nn.Module):
#     def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
#         super(Temporal_Attention_layer, self).__init__()
#         self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
#         self.U2 = nn.Parameter(torch.FloatTensor(
#             in_channels, num_of_vertices).to(DEVICE))
#         self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
#         self.be = nn.Parameter(torch.FloatTensor(
#             1, num_of_timesteps, num_of_timesteps).to(DEVICE))
#         self.Ve = nn.Parameter(torch.FloatTensor(
#             num_of_timesteps, num_of_timesteps).to(DEVICE))

#     def forward(self, x):
#         '''
#         :param x: (batch_size, N, F_in, T)
#         :return: (B, T, T)
#         '''
#         _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

#         lhs = torch.matmul(torch.matmul(
#             x.permute(0, 3, 2, 1), self.U1), self.U2)
#         # x:(B, N, F_in, T) -> (B, T, F_in, N)
#         # (B, T, F_in, N)(N) -> (B,T,F_in)
#         # (B,T,F_in)(F_in,N)->(B,T,N)

#         rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

#         product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

#         E = torch.matmul(self.Ve, torch.sigmoid(
#             product + self.be))  # (B, T, T)

#         E_normalized = F.softmax(E, dim=1)

#         return E_normalized

class Multi_Head_Temporal_Attention_layer(nn.Module):
    '''
    Multi-head temporal attention.
    Each head learns a different (T, T) time relationship map.
    Averaged at the end → still outputs (B, T, T).
    Drop-in replacement for Temporal_Attention_layer.
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps, num_heads=4):
        super(Multi_Head_Temporal_Attention_layer, self).__init__()
        self.num_heads = num_heads

        # each head gets its own set of parameters
        # same as spatial — num_heads separate students, each with own worksheet
        self.U1 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.U2 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.U3 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.be = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
            for _ in range(num_heads)
        ])
        self.Ve = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))
            for _ in range(num_heads)
        ])

    def forward(self, x):
        '''
        :param x: (B, N, F, T)
        :return:  (B, T, T)  ← same shape as before, drop-in replacement
        '''
        head_outputs = []

        for h in range(self.num_heads):
            # exact same math as original Temporal_Attention_layer
            lhs = torch.matmul(torch.matmul(
                x.permute(0, 3, 2, 1), self.U1[h]), self.U2[h])
            # (B,T,F,N)(N) -> (B,T,F) -> (B,T,N)

            rhs = torch.matmul(self.U3[h], x)  # (B,N,T)

            product = torch.matmul(lhs, rhs)    # (B,T,T)

            E = torch.matmul(self.Ve[h], torch.sigmoid(
                product + self.be[h]))           # (B,T,T)

            E_normalized = F.softmax(E, dim=1)

            head_outputs.append(E_normalized)   # each is (B, T, T)

        # stack → (B, H, T, T) then average → (B, T, T)
        all_heads = torch.stack(head_outputs, dim=1)
        merged = all_heads.mean(dim=1)

        return merged


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(
            in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(
                    0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps, num_heads=4):
        super(ASTGCN_block, self).__init__()
        self.TAt = Multi_Head_Temporal_Attention_layer(
            DEVICE, in_channels, num_of_vertices, num_of_timesteps, num_heads=num_heads)
        # self.SAt = Multi_Head_Spatial_Attention_layer(
        #     DEVICE, in_channels, num_of_vertices, num_of_timesteps, num_heads=num_heads)
        self.SAt = Spatial_Attention_layer(
            DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(
            K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(
            1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # print('='*10)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # print(x.mean())

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        # print(temporal_At.mean())


        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(
            batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # print(x_TAt.mean())
        # print('='*10)

        # SAt
        spatial_At = self.SAt(x_TAt)
        # print(spatial_At.mean())

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # print(spatial_gcn.mean())
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        # print(time_conv_output.mean())

        # residual shortcut
        # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        # print(x_residual.mean())

        x_residual = self.ln(
            F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual

class ASTGCN_MH(BaseModel):
    def __init__(self,
                 nb_block,
                 K,
                 nb_chev_filter,
                 nb_time_filter,
                 time_strides,
                 cheb_polynomials,
                 num_heads=4,
                 **args):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''
        super(ASTGCN_MH, self).__init__(**args)
        # self.embedding_air=AirEmbedding()
        self.BlockList = nn.ModuleList([ASTGCN_block(self.device, self.input_dim, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, self.num_nodes, self.seq_len, num_heads)])

        self.BlockList.extend([ASTGCN_block(
            self.device, nb_time_filter, K, nb_chev_filter, nb_time_filter,
            1, cheb_polynomials, self.num_nodes, self.seq_len//time_strides,
            num_heads) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(
            int(self.seq_len/time_strides), self.horizon, kernel_size=(1, nb_time_filter))
        
        # print(cheb_polynomials)


    def forward(self, inputs, supports=None):
        # x: [b, t, n, c]
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        ''' 
        # print(inputs.mean())

        inputs = inputs[..., :self.input_dim]
        x = inputs.permute(0, 2, 3, 1)
        # print(x.mean())

        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[
            :, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)


        return output.permute(0, 2, 1).unsqueeze(-1)


# def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices):
#     '''
#     :param DEVICE:
#     :param nb_block:
#     :param in_channels:
#     :param K:
#     :param nb_chev_filter:
#     :param nb_time_filter:
#     :param time_strides:
#     :param cheb_polynomials:
#     :param nb_predict_step:
#     :param len_input
#     :return:
#     '''
#     L_tilde = scaled_Laplacian(adj_mx)
#     cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(
#         DEVICE) for i in cheb_polynomial(L_tilde, K)]
#     model = ASTGCN(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
#                              time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)

#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#         else:
#             nn.init.uniform_(p)

#     return model
