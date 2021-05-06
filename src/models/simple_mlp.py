import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    """
    A simple feed-forward network with one input/output unit, two hidden layers and tanh activation.

    Attributes
    ----------
    hidden_size : int
        Number of hidden units per hidden layer
    hidden_layer_1 : str
        the name of the animal
    hidden_layer_2 : str
        the sound that the animal makes
    output_layer : int
        the number of legs the animal has (default 4)
    """

    def __init__(self, hidden_size=32):
        """
        Parameters
        ----------
        hidden_size : int
            Number of hidden units per hidden layer (default is 32)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_layer_1 = nn.Linear(1, self.hidden_size)
        self.hidden_layer_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer_4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        """Forward pass of the network.

        Parameters
        ----------
        x : float or numpy.array or torch.Tensor
            The network input

        Returns
        ----------
        torch.Tensor
            The network output in (batch_size, 1) shape
        """
        # convert input to torch.Tensor if necessary
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)

        # get input into shape (batch_size, 1)
        x = x.view((-1, 1))

        # compute forward pass
        z1 = torch.tanh(self.hidden_layer_1(x))
        z2 = torch.tanh(self.hidden_layer_2(z1))
        z3 = torch.tanh(self.hidden_layer_3(z2))
        z4 = torch.tanh(self.hidden_layer_3(z3))
        output = self.output_layer(z4)
        return output

    def predict(self, x):
        """Compute the network output for a given input and return the result as numpy array.

        Parameters
        ----------
        x : float or numpy.array or torch.Tensor
            The network input

        Returns
        ----------
        numpy.array
            The network output in (batch_size, 1) shape
        """
        output = self(x)
        return output.detach().numpy()

    def fit(self, data, targets, batch_size=5, learning_rate=5e-3, max_epochs=300):
        """Train the network on the given data.

        Parameters
        ----------
        data : numpy.array
            The input training samples
        targets : numpy.array
            The output training samples
        batch_size : int
            Batch size for mini-batch gradient descent training
        learning_rate : float
            The learning rate for the optimizer
        max_epochs : int
            Maximum number of training epochs

        Returns
        ----------
        list
            List of training set MSE scores for each training epoch
        """
        # define the loss function (MSE) and the optimizer (Adam)
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # create a data loader that takes care of batching our training set
        data = torch.FloatTensor(data[:, np.newaxis]) if len(
            data.shape) < 2 else torch.FloatTensor(data)
        targets = torch.FloatTensor(targets[:, np.newaxis]) if len(
            targets.shape) < 2 else torch.FloatTensor(targets)
        training_set = TensorDataset(data, targets)
        train_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True)

        # train the network
        train_loss_list = []
        for i in range(max_epochs):
            for batch in train_loader:
                batch_data = batch[0]
                batch_targets = batch[1]

                # compute batch loss
                predictions = self(batch_data)
                batch_loss = loss(predictions, batch_targets)

                # optimize the network parameters
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # record the training set MSE after each epoch; `with torch.no_grad()` makes sure that we do not record
            # gradient information of anything inside this environment
            with torch.no_grad():
                predictions = self(training_set[:][0])
                train_loss = loss(predictions, training_set[:][1])
                train_loss_list.append(train_loss.detach().numpy())

            # log the training loss
            print(f'Epoch {i} training loss is {train_loss:.2f}', end='\r')
            if i % 50 == 0:
                print('')

        return train_loss_list

    def save(self, path):
        """Save the network weights

        Parameters
        ----------
        path : str
            Target storage path
        """
        torch.save({
            'state_dict': self.state_dict(),
        }, path)

    def load(self, path):
        """Load the network weights from a file

        Parameters
        ----------
        path : str
            File containing the stored weights
        """
        model = torch.load(path)
        self.load_state_dict(model['state_dict'])
