# ------------------------------------------------------------------------------------------
# Defines the 3 parameter POC neural network architecture used to predict the kSZ
# angular power spectrum from reionization parameters.
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

import torch.nn as nn


class ThreeParamEmulator(nn.Module):
    def __init__(self):
        """
        Small neural network for predicting the log of the angular power spectrum
        from reionization parameters as a proof of concept.
        Input: Tensor of shape (N, 3) containing (alpha, kb, b0)
        Output: Tensor of shape (N, 5) containing log(D_ell) for 5 ell bins
        Architecture: 3 -> -> 5 -> 5 (GELU)
        """
        super().__init__()

        self.fc1 = nn.Linear(3, 5)
        self.out = nn.Linear(5, 5)

        self.activation = nn.GELU()


    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.out(x)

        return x

#-----------------------------
#         END OF FILE
#-----------------------------