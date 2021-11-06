import model
import abc


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import loss


class DVAEtraining(metaclass=abc.ABCMeta):
    """
    A strategy for training

    This is an abstract class meant to be inherited
    """

    @abc.abstractmethod
    def train(
            self,
            model: model.DVAEdatadeclaration
    ):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEtrainingNormal(DVAEtraining):
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
            model: model.DVAEdatadeclaration
    ):

        nn = model.create_model()

        optimizer = optim.Adam(nn.parameters(), lr=self.lr)

        #todo set up loader

        for i, (x_mb, y_mb) in enumerate(train_loader):
            optimizer.zero_grad()

            loss_recorder = loss.DVAEloss()

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
            training: DVAEtrainingNormal,
            percent_training: float
    ):
        self.training = training
        self.percent_training = percent_training
        # to dictionary of hyper parameters

        # some way of random subsetting train vs test

    def train(
            self,
            model: model.DVAEdatadeclaration
    ):
        # do grid optimization for the hyperparameters
        self.training.train(model)
