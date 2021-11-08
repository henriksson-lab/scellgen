import _dataloader
import core
import abc

import pytorch_model_summary as pms


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
            lr=1e-3,
            num_epoch: int = 100
    ):
        self.num_epoch = num_epoch
        self.lr = lr


    def train(
            self,
            mod: core.DVAEmodel
    ):
        do_optimize = True
        optimizer = optim.Adam(mod.parameters(), lr=self.lr)
        dataset = mod.get_dataset()
        dl = _dataloader.BatchSamplerLoader(dataset)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("====== num params {}".format(count_parameters(mod)))

        for cur_epoch in range(0,self.num_epoch):
            for i, minibatch_data in enumerate(dl):
                optimizer.zero_grad()

                print("training {} epoch {} batch {} ".format(self.__class__.__name__, cur_epoch, i))

                loss_recorder = mod.forward(minibatch_data)
                total_loss = loss_recorder.get_total_loss()
                print(total_loss)

                if do_optimize:
                    # Calculate gradients and improve values accordingly
                    total_loss.backward()
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
