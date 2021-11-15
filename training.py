import _dataloader
import core
import abc

import torch
import torch.optim as optim
import torch.utils.data


def get_torch_device():
    """
    Get a device - GPU if possible
    """
    print("CUDA is available: {}".format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Chose device: ".format(device))
    return device


def count_parameters(model):
    """
    Count the number of optimizable parameters. Note that this function might double count!
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    def __init__(
            self,
            lr=1e-3,
            num_epoch: int = 100
    ):
        """
        Your normal for-loop doing gradient descent...
        """
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

        print("====== num params {}".format(count_parameters(mod)))
        
        epoch_losses = []
        total_epoch_losses = []
        

        for cur_epoch in range(0, self.num_epoch):
            total_epoch_loss = 0
            all_losses = dict()
            for i, minibatch_data in enumerate(dl):
                optimizer.zero_grad()

                loss_recorder = mod.forward(minibatch_data, do_sampling=True)
                total_loss = loss_recorder.get_total_loss()
                # print(total_loss)
                total_epoch_loss += total_loss.cpu()

                all_losses = loss_recorder.add_losses(all_losses)

                if do_optimize:
                    # Calculate gradients and improve values accordingly
                    total_loss.backward()
                    optimizer.step()
            print("training {} epoch {} loss {}, in parts {}".format(self.__class__.__name__,
                                                                     cur_epoch, total_epoch_loss, str(all_losses)))
            epoch_losses.append(total_epoch_loss)
            total_epoch_losses.append(all_losses)
        return epoch_losses, total_epoch_losses
        
            


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEtrainingOptimizeHyperparameters(DVAEtraining):

    def __init__(
            self,
            training: DVAEtraining,
            percent_training: float
    ):
        """
        This optimizer will find suitable hyperparameters by cross validation.
        Inside it will run another optimizer for each hyperparameter value
        """
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
