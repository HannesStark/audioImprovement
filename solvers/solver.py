import torch
import matplotlib.pyplot as plt


class Solver():
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), create_plots=False):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.create_plots = create_plots

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_loss_history_per_iter = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            model.train()
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optim.step()

                if self.create_plots:
                    temp = outputs
                    plt.plot(labels[0][0], label='Ground truth')
                    plt.plot(temp[0][0].detach().numpy(), label='Prediction')
                    plt.ylim(-1.1, 1.1)
                    plt.legend(loc="upper right")
                    plt.show()

                loss_item = loss.item()
                if i % log_nth == log_nth - 1:
                    print("difference between inputs and predictions: "+ str(self.loss_func(inputs, outputs).item()))
                    print("difference between inputs and labels: " + str(self.loss_func(inputs, labels).item()))
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' %
                          (epoch + 1, i + 1, iter_per_epoch, loss_item))

                epoch_loss += loss_item
                self.train_loss_history_per_iter.append(loss_item)
                if i + 1 == iter_per_epoch:
                    self.train_loss_history.append(epoch_loss/iter_per_epoch)
                    print('[Epoch %d] Average loss of Epoch: %.7f' %
                          (epoch + 1, epoch_loss/iter_per_epoch))

            model.eval()
            val_loss = 0
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, labels)
                val_loss += loss.item()
                if i + 1 == len(val_loader):
                    val_loss /= i
                    self.val_loss_history.append(val_loss)
                    print('[Epoch %d] VAL loss: %.7f' %
                          (epoch + 1, val_loss))
        print('FINISH.')
