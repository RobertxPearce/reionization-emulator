# ------------------------------------------------------------------------------------------
# Experimental 4-parameter POC neural network variants for the kSZ angular power spectrum.
# For stable baseline use FourParamEmulator from reionemu.models.
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

import torch.nn as nn


class POCEmulatorFourParamsV1(nn.Module):
    def __init__(self):
        """
        POC NN for predicting the binned kSZ angular power spectrum from reionization parameters.
        Input: Tensor of shape (N, 4) containing (zmean, alpha, kb, b0)
        Output: Tensor of shape (N, 5) containing log(D_ell) for 5 ell bins
        Architecture: 4 -> 20 -> 5 (GELU)
        """
        super().__init__()

        self.fc1 = nn.Linear(4, 20)
        self.out = nn.Linear(20, 5)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.out(x)

        return x


class POCEmulatorFourParamsV2(nn.Module):
    def __init__(self):
        """
        POC NN for predicting the binned kSZ angular power spectrum from reionization parameters.
        Input: Tensor of shape (N, 4) containing (zmean, alpha, kb, b0)
        Output: Tensor of shape (N, 5) containing log(D_ell) for 5 ell bins
        Architecture: 4 -> 20 -> 20 -> 20 -> 5 (ReLU + Dropout=0.1)
        """
        super().__init__()

        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 5)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.out(x)

        return x


class POCEmulatorFourParamsV3(nn.Module):
    def __init__(self):
        """
        POC NN for predicting the binned kSZ angular power spectrum from reionization parameters.
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
