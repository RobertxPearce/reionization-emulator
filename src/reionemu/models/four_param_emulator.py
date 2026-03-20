# -----------------------------------------------------------------------------
# Defines the 4 parameter POC neural network architecture used to predict
# the kSZ angular power spectrum from reionization parameters.
#
# Robert Pearce
# -----------------------------------------------------------------------------

import torch.nn as nn


class FourParamEmulator(nn.Module):
    def __init__(self):
        """
        POC NN for predicting the binned kSZ angular power spectrum from reionization
        parameters.
        Input: Tensor of shape (N, 4) containing (zmean, alpha, kb, b0)
        Output: Tensor of shape (N, 5) containing log(D_ell) for 5 ell bins
        Architecture: 4 -> 20 -> 20 -> 5 (ReLU)
        """
        super().__init__()

        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 5)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out(x)

        return x


# -----------------------------
#         END OF FILE
# -----------------------------
