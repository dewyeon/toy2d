import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
from models.generative_flow import GenerativeFlow
import models.transformations as flows



class AffineFlow(GenerativeFlow):
    """
    Variational auto-encoder with affine flows in the encoder.
    """

    def __init__(self, args):
        super(AffineFlow, self).__init__(args)

        self.num_coefs = 2  # scale and shift 
        self.flow_coef = nn.Parameter(torch.randn(self.num_flows, self.z_size, self.num_coefs).normal_(0, 0.01))
        self.flow_transformation = flows.Affine()

    def flow(self, z_0):
        return self.forward(z_0)

    def reverse(self, z_0):
        raise NotImplementedError("Inverse flow not available yet")

    def forward(self, z_0):
        log_det_jacobian = 0.0
        z = [z_0]

        for k in range(self.num_flows):
            bs = z_0.size(0)
            flow_coef = self.flow_coef[k,...].expand(bs, self.z_size, self.num_coefs)
            z_k, ldj = self.flow_transformation(z[k], flow_coef)
            z.append(z_k)
            log_det_jacobian += ldj
        
        return z[-1], log_det_jacobian


class AffineVAE(VAE):
    """
    Variational auto-encoder with affine flows in the encoder.
    """

    def __init__(self, args):
        super(AffineVAE, self).__init__(args)
        self.num_flows = args.num_flows
        self.density_evaluation = args.density_evaluation
        self.num_coefs = 2  # scale and shift 
        
        # Amortized flow parameters
        if args.density_evaluation:
            # only performing an evaluation of flow, init flow parameters randomly
            self.flow_coef = nn.Parameter(torch.randn(self.num_flows, self.z_size, self.num_coefs).normal_(0, 0.01))

        else:
            # learned from encoder neural network
            self.amor_flow_coef = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_coefs)
            self.flow_coef = None

        # Normalizing flow layers
        self.flow_transformation = flows.Affine()

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        z_mu = self.q_z_mean(h)
        z_var = self.q_z_var(h)

        # compute amortized parameters for flows
        self.flow_coef = self.amor_flow_coef(h).view(batch_size, self.num_flows, self.z_size, self.num_coefs)

        return z_mu, z_var

    def flow(self, z_0):
        # Initialize log-det-jacobian to zero
        log_det_jacobian = 0.0
        z = [z_0]

        for k in range(self.num_flows):
            if self.density_evaluation:
                bs = z_0.size(0)
                flow_coef = self.flow_coef[k,...].expand(bs, self.z_size, self.num_coefs)
            else:
                flow_coef = self.flow_coef[:, k, :, :]
            
            z_k, ldj = self.flow_transformation(z[k], flow_coef)
            z.append(z_k)
            log_det_jacobian += ldj
        
        return z[-1], log_det_jacobian

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        z_mu, z_var = self.encode(x)

        # Sample z_0
        z_0 = self.reparameterize(z_mu, z_var)

        # pass through normalizing flow
        z_k, log_det_jacobian = self.flow(z_0)

        # reconstruct
        x_recon = self.decode(z_k)

        return x_recon, z_mu, z_var, log_det_jacobian, z_0, z_k

