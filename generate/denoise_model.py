from utils.typing_utils import *
from gnn import MLPBlock
from utils.model_utils import create_activation, create_normalization

class DenoiseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        norm: Optional[str] = 'bn',
        activation: Optional[str] = 'relu',
        dropout: float = 0.3
    ):
        r"""
        Denoise model for a given diffused graph. Construct of two mlp layer:
        mlp_noise(get the time embedding) + mlp_edge(get the edge probs)

        Args:
            input_dim:  the dimension of input tensor for mlp_edge
            hidden_dim: the dimension of hidden layer for mlp_noise and mlp_edge
            output_dim: the dimension of output tensor for mlp_noise
            norm:       the name of normalization method which will be used in denoise model
            activation: the name of activation method which will be used in denoise model
            dropout:    the coefficient of dropout act
        """
        super(DenoiseNet, self).__init__()
        # input_dim = 2 * input_dim + output_dim
        input_dim = 2 * input_dim + 1
        # main part
        # self.mlp_noise = MLPBlock(1, hidden_dim, output_dim, norm, activation)
        self.mlp_noise = MLPBlock(1, 4, 1, norm, activation, out_activation=None)
        self.mlp_edge  = MLPBlock(input_dim, hidden_dim, 1, norm, activation, out_activation=None)

        # other part
        self.dropout       = torch.nn.Dropout(p=dropout)
        # self.activation    = create_activation(activation)
        self.input_norm    = create_normalization(norm)(1)
        self.output_norm   = create_normalization(norm)(input_dim)

    def forward(
        self,
        batch_noise_edge: Tensor,
        batch_edge_index: Tensor,
        batch_embeddings: Tensor,
        batch: Tensor
    ):
        r"""
        Given embeddings and noise_schedule for per edges, compute new edge probs for generated graph.

        Args:
            batch_noise_edge: the noise level for per edges of graph in batch from 0 to t,
                                  i.e., \prod_{i = 0}^t \beta_i
            batch_edge_index:     adj of pyg data for a batch
            batch_embeddings:     embeddings of currant batch graph with given GNN model
            batch:                which graph belonged for every node

        Returns: preds without reparameterization, i.e., \beta_{t - 1}
        """
        if batch_noise_edge.dim() < 2:
            batch_noise_edge = batch_noise_edge[:, None]
        # compute embedding of batch_noise_schedule
        norm_batch_noise_edge = self.input_norm(batch_noise_edge)
        batch_noise_embeds    = self.mlp_noise(norm_batch_noise_edge)
        # create input for mlp_edge
        row, col             = batch_edge_index
        batch_edge_embeds    = torch.cat(
            [batch_embeddings[row], batch_embeddings[col], batch_noise_embeds], dim=-1
        )
        # compute output with mlp_edge
        batch_edge_embeds = self.output_norm(batch_edge_embeds)
        batch_edge_embeds = self.dropout(batch_edge_embeds)
        preds             = self.mlp_edge(batch_edge_embeds).view(-1)
        return preds