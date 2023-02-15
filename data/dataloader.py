from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='diffusercam')
class DiffuserDataset(VisionDataset):
    """Diffuser dataset https://waller-lab.github.io/LenslessLearning/dataset.html"""

    def __init__(self, csv_file, data_dir, label_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the Diffuser images.
            label_dir (string): Directory with all the natural images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_contents = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.csv_contents)

    def __getitem__(self, idx):

        img_name = self.csv_contents.iloc[idx,0]

        path_diffuser = os.path.join(self.data_dir, img_name) 
        path_gt = os.path.join(self.label_dir, img_name)
        
        image = np.load(path_diffuser+'.npy')
        label = np.load(path_gt+'.npy')

        image = image[:-58,62:-18,:]
        label = label[:-58,62:-18,:]

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return label, image