from imports import *
from utils import *

class UNET(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, side):

        super(UNET, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.side = side

        self.pool = nn.MaxPool3d(2, stride=2, return_indices=True)

        self.unpool = nn.MaxUnpool3d(2)

        self.fc1 = nn.Sequential(
        nn.Conv3d(input_dim, hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(hidden_dim),
        nn.SiLU(inplace=True),
        nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(hidden_dim),
        )

        self.fc2 = nn.Sequential(
        nn.Conv3d(hidden_dim, 2*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(2*hidden_dim),
        nn.SiLU(inplace=True),
        nn.Conv3d(2*hidden_dim, 2*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(2*hidden_dim),
        )

        self.fc3 = nn.Sequential(
        nn.Conv3d(2*hidden_dim, 4*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(4*hidden_dim),
        nn.SiLU(inplace=True),
        nn.Conv3d(4*hidden_dim, 4*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(4*hidden_dim),
        )

        self.fc4 = nn.Sequential(
        nn.Conv3d(4*hidden_dim, 8*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(8*hidden_dim),
        nn.SiLU(inplace=True),
        nn.Conv3d(8*hidden_dim, 4*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        )

        self.fc5 = nn.Sequential(
        nn.Conv3d(8*hidden_dim, 4*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(4*hidden_dim),
        nn.SiLU(inplace=True),
        nn.Conv3d(4*hidden_dim, 2*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        )

        self.fc6 = nn.Sequential(
        nn.Conv3d(4*hidden_dim, 2*hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(2*hidden_dim),
        nn.SiLU(inplace=True),
        nn.Conv3d(2*hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        )

        self.fc7 = nn.Sequential(
        nn.Conv3d(2*hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        nn.BatchNorm3d(hidden_dim),
        nn.SiLU(inplace=True),
        nn.Conv3d(hidden_dim, output_dim, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
        )

    def reset_parameters(self):

        for layer in self.fc1[:2] and self.fc1[3:]:
            layer.reset_parameters()
            
        for layer in self.fc2[:2] and self.fc2[3:]:
            layer.reset_parameters()

        for layer in self.fc3[:2] and self.fc3[3:]:
            layer.reset_parameters()

        for layer in self.fc4[:2] and self.fc4[3:]:
            layer.reset_parameters()

        for layer in self.fc5[:2] and self.fc5[3:]:
            layer.reset_parameters()

        for layer in self.fc6[:2] and self.fc6[3:]:
            layer.reset_parameters()

        for layer in self.fc7[:2] and self.fc7[3:]:
            layer.reset_parameters()

    def forward(self, x):
        x = torch.transpose(x.reshape(self.batch_size,self.side**3,self.input_dim), 1, 2).reshape(self.batch_size,self.input_dim,self.side,self.side,self.side)
        x_0 = F.silu(self.fc1(x))
        x_1, indices_1 = self.pool(x_0)
        x_2 = F.silu(self.fc2(x_1))
        x_3, indices_3 = self.pool(x_2)
        x_4 = F.silu(self.fc3(x_3))
        x_5, indices_5 = self.pool(x_4)
        x_6 = self.fc4(x_5)
        x_7 = torch.add(x_6,x_5)
        x_8 = self.unpool(x_7, indices_5)
        x_9 = torch.cat((x_8,x_4), axis=1)
        x_10 = self.fc5(x_9)
        x_11 = torch.add(x_10,x_3)
        x_12 = self.unpool(x_11, indices_3)
        x_13 = torch.cat((x_12,x_2), axis=1)
        x_14 = self.fc6(x_13)
        x_15 = torch.add(x_14,x_1)
        x_16 = self.unpool(x_15, indices_1)
        x_17 = torch.cat((x_16,x_0), axis=1)
        x_18 = self.fc7(x_17)
        out = torch.add(x_18,x)

        return out
    
class ViG(torch.nn.Module):
    def __init__(self, batch, side, input_dim, hidden_dim, output_dim, dropout):

        super(ViG, self).__init__()

        self.batch = batch
        self.side = side
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ENCODER

        self.cnn_1 = nn.Sequential(nn.Conv3d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2), nn.BatchNorm3d(hidden_dim,),)

        self.ginconv_1 = GINConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.bn_1 = torch.nn.BatchNorm1d(num_features=hidden_dim) 
        self.sageconv_1 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.bn_2 = torch.nn.BatchNorm1d(num_features=hidden_dim)
        self.sageconv_2 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.bn_3 = torch.nn.BatchNorm1d(num_features=hidden_dim) 

        self.linear_1 = torch.nn.Linear(hidden_dim, hidden_dim*2)
        self.linear_2 = torch.nn.Linear(hidden_dim*2, hidden_dim)

        self.cnn_2 = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2), nn.BatchNorm3d(hidden_dim,),)

        self.ginconv_2 = GINConv(Sequential(Linear(hidden_dim, hidden_dim*2), ReLU(), Linear(hidden_dim*2, hidden_dim*2)))
        self.bn_4 = torch.nn.BatchNorm1d(num_features=hidden_dim*2)
        self.sageconv_3 = SAGEConv(in_channels=hidden_dim*2, out_channels=hidden_dim*2)
        self.bn_5 = torch.nn.BatchNorm1d(num_features=hidden_dim*2)
        self.sageconv_4 = SAGEConv(in_channels=hidden_dim*2, out_channels=hidden_dim)
        self.bn_6 = torch.nn.BatchNorm1d(num_features=hidden_dim) 

        self.linear_3 = torch.nn.Linear(hidden_dim, hidden_dim*4)
        self.linear_4 = torch.nn.Linear(hidden_dim*4, hidden_dim)

        # DECODER

        self.linear_final = torch.nn.Linear(hidden_dim, output_dim)

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element to be zeroed-out
        self.dropout = dropout

    def reset_parameters(self):

        for layer in self.cnn_1:
            layer.reset_parameters()

        self.ginconv_1.reset_parameters()
        self.bn_1.reset_parameters()
        self.sageconv_2.reset_parameters()
        self.bn_2.reset_parameters()
        self.sageconv_3.reset_parameters()
        self.bn_3.reset_parameters()

        self.linear_1.reset_parameters()
        self.linear_2.reset_parameters()

        for layer in self.cnn_2:
            layer.reset_parameters()

        self.ginconv_2.reset_parameters()
        self.bn_4.reset_parameters()
        self.sageconv_3.reset_parameters()
        self.bn_5.reset_parameters()
        self.sageconv_4.reset_parameters()
        self.bn_6.reset_parameters()

        self.linear_3.reset_parameters()
        self.linear_4.reset_parameters()
        
        self.linear_final.reset_parameters()


    def forward(self, x, adj_t):

        #ENCODER
       
        x = torch.transpose(x.reshape(self.batch,self.side**3,self.input_dim), 1, 2).reshape(self.batch,self.input_dim,self.side,self.side,self.side)
        x = F.silu(self.cnn_1(x))
        x = torch.transpose(x.reshape(self.batch,self.hidden_dim,self.side**3),2,1).reshape(self.batch*(self.side**3),self.hidden_dim)

        id = x
        
        x = F.silu(self.bn_1(self.ginconv_1(x, adj_t)))
        x = F.silu(self.bn_2(self.sageconv_1(x, adj_t)))
        x = F.silu(self.bn_3(self.sageconv_2(x, adj_t)))

        x = x + id # Skip connection
        id = x 

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.silu(self.linear_1(x))
        x = F.silu(self.linear_2(x))

        x = x + id # Skip connection

        x = torch.transpose(x.reshape(self.batch,self.side**3,self.hidden_dim), 1, 2).reshape(self.batch,self.hidden_dim,self.side,self.side,self.side)
        x = F.silu(self.cnn_2(x))
        x = torch.transpose(x.reshape(self.batch,self.hidden_dim,self.side**3),2,1).reshape(self.batch*(self.side**3),self.hidden_dim)

        id = x 

        x = F.silu(self.bn_4(self.ginconv_2(x, adj_t)))
        x = F.silu(self.bn_5(self.sageconv_3(x, adj_t)))
        x = F.silu(self.bn_6(self.sageconv_4(x, adj_t)))

        x = x + id # Skip connection
        id = x 

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.silu(self.linear_3(x))
        x = F.silu(self.linear_4(x))

        x = x + id # Skip connection

        # DECODER

        out = self.linear_final(x)

        return out

class GNN(torch.nn.Module):
    def __init__(self, head, input_dim, hidden_dim, output_dim, dropout):

        super(GNN, self).__init__()

        # A list of GCNConv layers. self.convs has num_layers GCNConv layers
        self.gatconv = GATConv(in_channels=input_dim, out_channels=hidden_dim//2, heads=head)
        self.sageconv = SAGEConv(in_channels=head*hidden_dim//2, out_channels=hidden_dim)
        self.gcnconv = GCNConv(in_channels=hidden_dim, out_channels=2*hidden_dim)

        self.linear1 = torch.nn.Linear(2*hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()
        # Probability of an element to be zeroed
        self.dropout = dropout

    def reset_parameters(self):
        self.gatconv.reset_parameters()
        self.sageconv.reset_parameters()
        self.gcnconv.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.gatconv(x, edge_index))
        x = F.dropout(F.tanh(self.sageconv(x, edge_index)), p=self.dropout, training=self.training)
        x = F.leaky_relu(self.gcnconv(x, edge_index))
        x = F.dropout(F.elu(self.linear1(x)), p=self.dropout, training=self.training)
        out = self.linear2(x)

        return out
