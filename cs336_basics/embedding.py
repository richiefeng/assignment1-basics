import torch
import torch.nn as nn
import torch.nn.init as init


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.
        
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        # Create embedding matrix parameter with shape (vocab_size, d_model)
        # Store with d_model as the final dimension as specified
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights using trunc_normal_ with std=0.02
        init.trunc_normal_(self.embedding_matrix, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids (torch.Tensor): Tensor of token IDs with shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Embedding vectors with shape (batch_size, sequence_length, embedding_dim)
        """
        # Use advanced indexing to select embedding vectors for each token ID
        # token_ids shape: (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        return self.embedding_matrix[token_ids]