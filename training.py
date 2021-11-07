import core
import abc


import torch
import torch.optim as optim
import torch.utils.data


def get_torch_device():
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    return device


class DVAEtraining(metaclass=abc.ABCMeta):
    """
    A strategy for training

    This is an abstract class meant to be inherited
    """

    @abc.abstractmethod
    def train(
            self,
            mod: core.DVAEmodel
    ):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEtrainingBasic(DVAEtraining):
    """
    Your normal for-loop doing gradient descent...
    """

    def __init__(
            self,
            lr=1e-3
            # todo learning rate. num epochs. what else?
    ):
        self.lr = lr

    def train(
            self,
            mod: core.DVAEmodel
    ):
        do_optimize = True
        optimizer = optim.Adam(mod.parameters(), lr=self.lr)
        dataloader = mod.get_dataloader()

        for i, minibatch_data in enumerate(dataloader):
            optimizer.zero_grad()

            loss_recorder = mod.forward(minibatch_data)

            if do_optimize:
                # Calculate gradients and improve values accordingly
                loss_recorder.get_total_loss().backward()
                optimizer.step()





######################################################################################################
######################################################################################################
######################################################################################################

class DVAEtrainingOptimizeHyperparameters(DVAEtraining):
    """
    This optimizer will find suitable hyperparameters by cross validation.
    Inside it will run another optimizer for each hyperparameter value
    """

    def __init__(
            self,
            training: DVAEtraining,
            percent_training: float
    ):
        self.training = training
        self.percent_training = percent_training
        # to dictionary of hyper parameters

        # some way of random subsetting train vs test

    def train(
            self,
            mod: core.DVAEmodel
    ):
        # do grid optimization for the hyperparameters
        self.training.train(mod)
