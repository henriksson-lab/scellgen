
import datadecl


class DVAEtraining():
    """
    A strategy for training

    This is an abstract class meant to be inherited
    """

    def train(
            self,
            model: datadecl.DVAEdatadeclaration
    ):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEtrainingNormal(DVAEtraining):
    """
    Your normal for-loop doing gradient descent...
    """


    def train(
            self,
            model: datadecl.DVAEdatadeclaration
    ):
        666
        # todo




