# ------------------------------------------------------------------------------------------
# Defines the neural network architecture used to predict the kSZ angular power
# spectrum from reionization parameters.
# Robert Pearce
# ------------------------------------------------------------------------------------------

import torch.nn as nn


class ProofOfConceptEmulatorThreeParams(nn.Module):
    def __init__(self):
        """
        Small neural network emulator for predicting the log of the angular power spectrum
        from reionization parameters as a proof of concept.
        Architecture:
            Input: 3 parameters (zmean, alpha, kb, b0 is const)
            Hidden: 5
            Output: 5 log(d_ell) values (one per ell-bin)
        """
        super().__init__()

        # Single hidden layer: 3 -> 5
        self.fc1 = nn.Linear(3, 5)

        # Output layer: 5 -> 5
        self.out = nn.Linear(5, 5)

        # Activation function for hidden layers
        self.activation = nn.GELU()


    def forward(self, x):
        """
        Forward pass through the neural network.
        Input: 3 parameters (zmean, alpha, kb, b0 is const)
        Output: 5 log(d_ell) values (one per ell-bin)
        """
        x = self.activation(self.fc1(x))
        x = self.out(x)

        return x

#-----------------------------
#         END OF FILE
#-----------------------------
