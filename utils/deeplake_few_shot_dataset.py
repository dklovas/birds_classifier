from PIL import Image
import torch


class DeepLakeFewShotDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for Few-Shot Learning with DeepLake data.

    This class represents a custom dataset for few-shot learning, compatible with PyTorch's
    `DataLoader` for use in training, validation, or testing. It retrieves images and labels
    from the provided dataset and applies optional transformations. It also provides a method
    to get the labels of the dataset.

    Args:
        ds (Dataset): The dataset containing the images and labels.
        transform (callable, optional): A function/transform to apply to the images.

    Methods:
        __getitem__(idx): Retrieves an image and its corresponding label at index `idx`.
        __len__(): Returns the total number of samples in the dataset.
        get_labels(): Returns a list of all the labels in the dataset.
    """

    def __init__(self, ds, transform=None):
        """
        Initializes the DeepLakeFewShotDataset instance.

        Args:
            ds (Dataset): The dataset containing the images and labels.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.ds = ds
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves an image and its corresponding label at index `idx`.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image, label, and index.
        """
        image = self.ds[idx]["images"].numpy()
        label = self.ds[idx]["labels"].numpy()
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, int(label), idx

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.ds)

    def get_labels(self) -> list:
        """
        Returns a list of all the labels in the dataset.

        Returns:
            list: A list of labels for each sample in the dataset.
        """
        return [int(label) for label in self.ds["labels"].numpy()]
