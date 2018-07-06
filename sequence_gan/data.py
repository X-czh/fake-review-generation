import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class YelpDataSet(Dataset):

    def __init__(self, csv_file, img_path, img_ext, transform):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 0

def get_dataloader():
    train_loader = DataLoader(
        dataset=partition,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1)
    return train_loader
