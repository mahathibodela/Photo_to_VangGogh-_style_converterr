from torch.utils.data import Dataset
from PIL import Image
from config import Config
config = Config()


class PhotoToVanggogfDataset(Dataset):
    def __init__(self, vanggogf_photos, real_photo, transform:bool = True)->None:
        self.vanggohf_photos = vanggogf_photos
        self.real_photos = real_photos
        self.transform = transform

    def __len__(self)->int:
        return max(len(self.vanggohf_photos), len(self.real_photos))

    def __getitem__(self, idx)->tuple:
        vanggogf_img = Image.open(self.self.vanggohf_photos[idx%len(self.vanggohf_photos)])
        real_img = Image.open(self.real_photos[idx%len(self.real_photos)])
        if self.transform:
            real_img = config.preprocess(real_img)
            vanggogf_img = config.preprocess(vanggogf_img)

        return real_img, vanggogf_img