#!/usr/bin/env bash
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_dir_L=4, hidden_layer_size=256):
        super(NeRFmodel, self).__init__()

        # Calculate input dimensions with encoding
        self.input_dim_pos = 3 + 3 * 2 * embed_pos_L
        self.input_dim_dir = 3 + 3 * 2 * embed_dir_L

        # Define the MLP
        self.layers = nn.ModuleList()
        for layer_index in range(8):
            # Define the layers according to the provided filter size and input dimension
            if layer_index == 0:
                input_features = self.input_dim_pos 
            else:
                input_features = hidden_layer_size
            if layer_index in [4]:
                input_features += self.input_dim_pos
            if layer_index in [7]:
                output_features = hidden_layer_size + 1
            else:
                output_features = hidden_layer_size
            # Append the layer to the list
            self.layers.append(nn.Linear(input_features, output_features))
        # Feature layer
        self.feat_layer = nn.Linear(hidden_layer_size + self.input_dim_dir, hidden_layer_size // 2)
        # Output layer
        self.rgb_layer = nn.Linear(hidden_layer_size // 2, 3)

        # Store the positional encoding length
        self.embed_pos_L = embed_pos_L
        self.embed_dir_L = embed_dir_L

    def position_encoding(self, inputs, levels):
        # Encode the input with positional encoding
        encoded = [inputs]
        # Iterate over the levels
        for level in range(levels):
            # Calculate the scale
            encoded.append(torch.sin(2**level * np.pi * inputs))
            encoded.append(torch.cos(2**level * np.pi * inputs))

        output_encoding = torch.cat(encoded, dim=-1)
        return output_encoding

    def forward(self, positions, directions):
        # Encode position input
        encoded_positions = self.position_encoding(positions, self.embed_pos_L)
        # Initialize feature tensor
        for layer_index, layer in enumerate(self.layers):
            # Skip connection
            if layer_index in [4] and layer_index > 0:
                # Concatenate the input with the output of the first layer
                encoded_positions = torch.cat([encoded_positions, self.position_encoding(positions, self.embed_pos_L)], -1)
            encoded_positions = F.relu(layer(encoded_positions))
        # Extract sigma
        sigma, encoded_positions = encoded_positions[..., -1], encoded_positions[..., :-1]
        # Encode direction input
        encoded_directions = torch.cat([encoded_positions, self.position_encoding(directions, self.embed_dir_L)], -1)
        encoded_directions = F.relu(self.feat_layer(encoded_directions))
        # Output layer
        rgb_values = self.rgb_layer(encoded_directions)

        return F.sigmoid(rgb_values), sigma
