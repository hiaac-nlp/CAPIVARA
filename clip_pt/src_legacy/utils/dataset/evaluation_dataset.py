import abc

from torch.utils.data.dataset import Dataset


class EvaluationDataset(Dataset):
    @abc.abstractmethod
    def get_labels(self):
        pass
