import model
import abc


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
            self
            # todo learning rate. num epochs. what else?
    ):
        666

    def train(
            self,
            model: model.DVAEdatadeclaration
    ):
        666
        # todo


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
