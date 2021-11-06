import torch
import functools

class DVAEloss():

    def __init__(self):
        self.losses = dict()

    def add(self,
            category: str,
            loss: torch.Tensor
            ):
        """
        Add a named loss
        """
        if category in self.losses:
            self.losses[category] = self.losses[category] + loss
        else:
            self.losses[category] = loss

    def add_kl(self, loss: torch.Tensor):
        """
        Add a KL-loss
        """
        self.add("kl", loss)


    def get_total_loss(self):
        """
        Get the total recorded loss
        """
        assert bool(self.losses), "No losses have been recorded"
        return functools.reduce(lambda x, y: x+y, list(self.losses.values()))





