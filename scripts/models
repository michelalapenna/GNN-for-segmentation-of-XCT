from imports import *

batch = 8

class GNN(torch.nn.Module):
    def __init__(self, head, input_dim, hidden_dim, output_dim, dropout):

        super(GNN, self).__init__()

        # A list of GCNConv layers. self.convs has num_layers GCNConv layers
        self.gatconv = GATConv(in_channels=input_dim, out_channels=hidden_dim//2, heads=head)
        #self.ginconv = GINConv(Sequential(Linear(head*150, 180), ReLU(), Linear(180, 240)))
        self.sageconv = SAGEConv(in_channels=head*hidden_dim//2, out_channels=hidden_dim)
        self.gcnconv = GCNConv(in_channels=hidden_dim, out_channels=2*hidden_dim)

        self.linear1 = torch.nn.Linear(2*hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()
        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        #self.return_embeds = return_embeds

    def reset_parameters(self):
        self.gatconv.reset_parameters()
        #self.ginconv.reset_parameters()
        self.sageconv.reset_parameters()
        self.gcnconv.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x, edge_index, batch_index):
        x = F.relu(self.gatconv(x, edge_index))
        #x = F.tanh(self.ginconv(x, edge_index))
        x = F.dropout(F.tanh(self.sageconv(x, edge_index)), p=self.dropout, training=self.training)
        x = F.leaky_relu(self.gcnconv(x, edge_index))
        x = F.dropout(F.elu(self.linear1(x)), p=self.dropout, training=self.training)
        out = self.linear2(x)

        return out
