# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        print('Note: no longer zero-init last BNs by default')
        self.encoder = base_encoder(num_classes=dim)
        self._build_projector_and_predictor_mlps(dim, pred_dim)

    # def forward(self, x1, x2):
    #     """
    #     Input:
    #         x1: first views of images
    #         x2: second views of images
    #     Output:
    #         p1, p2, z1, z2: predictors and targets of the network
    #         See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
    #     """
    #
    #     # compute features for one view
    #     z1 = self.encoder(x1)  # NxC
    #     z2 = self.encoder(x2)  # NxC
    #
    #     p1 = self.predictor(z1)  # NxC
    #     p2 = self.predictor(z2)  # NxC
    #
    #     return p1, p2, z1.detach(), z2.detach()

    def _build_projector_and_predictor_mlps(self, dim, pred_dim):
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC

        p1 = self.predictor(z1)  # NxC

        return p1, z1.detach()

    # def get_feature_and_pred(self, x):
    #     z = self.encoder(x)
    #     p = self.predictor(z)
    #     return z.detach(), p.detach()


class SimSiamViT(SimSiam):
    def _build_projector_and_predictor_mlps(self, dim, pred_dim):
        prev_dim = self.encoder.head.weight.shape[1]
        del self.encoder.head

        # projectors
        self.encoder.head = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                          nn.BatchNorm1d(prev_dim),
                                          nn.ReLU(inplace=True),  # first layer
                                          nn.Linear(prev_dim, prev_dim, bias=False),
                                          nn.BatchNorm1d(prev_dim),
                                          nn.ReLU(inplace=True),  # second layer
                                          nn.Linear(prev_dim, dim),
                                          nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.head[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer


class NNSiam(SimSiam):
    """
    Build a Nearest Neighbors SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512, queue_length=12800 * 2):
        super().__init__(base_encoder, dim, pred_dim)
        self.register_buffer("queue", torch.randn(queue_length, dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def find_nn(self, features):
        with torch.no_grad():
            distances = l2(features, self.queue)
            idxs = torch.argmin(distances, dim=1)
            return self.queue[idxs].detach()

    def update_queue(self, features):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        assert len(self.queue) % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = features
        ptr = (ptr + batch_size) % len(self.queue)  # move pointer
        self.queue_ptr[0] = ptr


def l2(x1, x2):
    # squared l2 norm
    x1_norm = (x1 ** 2).sum(dim=1)
    x2_norm = (x2 ** 2).sum(dim=1)
    cross_term = -2 * torch.einsum('ik,jk->ij', [x1, x2])

    # return torch.linalg.norm(x1.unsqueeze(1) - x2.unsqueeze(0), dim=2)
    return x1_norm.unsqueeze(1) + x2_norm.unsqueeze(0) + cross_term
