from pathlib import Path
from torch.utils.data import Dataset
import torchvision
from glob import glob
import cv2
import numpy as np
import random


class Base(Dataset):
    def __init__(
        self,
        path,
        image_folder,
        dmap_folder,
        transforms=None,
        ext=".jpg",
        dmap_ext=".npy",
    ):
        super().__init__()
        self.path = path
        self.image_fnames = glob(self.path + "/" + image_folder + "/*" + ext)
        self.dmap_folder = path + "/" + dmap_folder
        self.transforms = transforms
        self.dmap_ext = dmap_ext

    def __len__(self):
        return len(self.image_fnames)

    def _get_dmap_name(self, fn):
        mask_name = fn.split("/")[-1].split(".")[0]  # basename
        mask_path = self.dmap_folder + "/" + mask_name + self.dmap_ext
        return mask_path

    def _load_image_and_density_map(self, idx):
        image_fname = self.image_fnames[idx]
        dmap_fname = self._get_dmap_name(image_fname)
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0
        d_map = np.load(dmap_fname, allow_pickle=True)

        return image, d_map


class Crop_Dataset(Base):
    def __init__(
        self,
        path,
        image_folder,
        dmap_folder,
        transforms=None,
        crop_size=512,
        ext=".jpg",
        dmap_ext=".npy",
        method="train",
        LOG_PARA=1000,
    ):
        super().__init__(
            path, image_folder, dmap_folder, transforms, ext, dmap_ext=dmap_ext
        )
        self.crop_size = crop_size
        if method not in ["train", "valid"]:
            raise Exception("Not Implement")
        self.method = method
        self.log_para = LOG_PARA

    def __getitem__(self, idx):
        fn = self.image_fnames[idx]
        image, density_map = self._load_image_and_density_map(idx)
        h, w = image.shape[0], image.shape[1]

        if self.method == "train":
            if w < self.crop_size or h < self.crop_size:
                pad_top = pad_bot = pad_left = pad_right = 0
                if w < self.crop_size:
                    diff = self.crop_size - w
                    if diff % 2 == 0:
                        pad_left = pad_right = diff // 2
                    else:
                        pad_left = diff // 2
                        pad_right = pad_left + 1
                if h < self.crop_size:
                    diff = self.crop_size - h
                    if diff % 2 == 0:
                        pad_top = pad_bot = diff // 2
                    else:
                        pad_top = diff // 2
                        pad_bot = pad_top + 1
                image = np.pad(
                    image,
                    ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                density_map = np.pad(
                    density_map,
                    ((pad_top, pad_bot), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )
            h, w = image.shape[:2]
            i, j = self._random_crop(h, w, self.crop_size, self.crop_size)
            image = image[i : i + self.crop_size, j : j + self.crop_size]
            density_map = density_map[i : i + self.crop_size, j : j + self.crop_size]
        else:
            pad_h, pad_w, pad_h_even, pad_w_even = self._get_pad((h, w), self.crop_size)
            if pad_h_even:
                pad_h_left = pad_h_right = pad_h // 2
            else:
                pad_h_left = pad_h // 2
                pad_h_right = pad_h_left + 1
            if pad_w_even:
                pad_w_left = pad_w_right = pad_w // 2
            else:
                pad_w_left = pad_w // 2
                pad_w_right = pad_w_left + 1
            image = np.pad(
                image,
                ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right), (0, 0)),
                "constant",
            )
            density_map = np.pad(
                density_map,
                ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right)),
                "constant",
            )
        if self.transforms:
            for tfms in self.transforms:
                aug = tfms(
                    **{
                        "image": image,
                        "mask": density_map,
                    }
                )
                image, density_map = aug["image"], aug["mask"]
        return image, density_map * self.log_para, self.image_fnames[idx]

    def _random_crop(self, im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j

    def _get_pad(self, im_sz, crop_sz=512, **kwargs):
        h, w = im_sz
        h_mul = h // crop_sz + 1
        w_mul = w // crop_sz + 1
        pad_h = crop_sz * h_mul - h
        pad_w = crop_sz * w_mul - w
        return pad_h // 2, pad_w // 2, pad_h % 2 == 0, pad_w % 2 == 0


class Crop_Dataset_Infer:
    def __init__(
        self, image_folder, transforms=None, crop_size=512, ext=".jpg", LOG_PARA=1000
    ):
        self.crop_size = crop_size

        self.log_para = LOG_PARA
        self.image_fnames = list(
            map(lambda x: x.as_posix(), Path(image_folder).rglob(f"*{ext}"))
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.image_fnames)

    def _load_image(self, idx):
        image_fname = self.image_fnames[idx]
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0
        return image

    def __getitem__(self, idx):
        image = self._load_image(idx)
        h, w = image.shape[0], image.shape[1]

        pad_h, pad_w, pad_h_even, pad_w_even = self._get_pad((h, w), self.crop_size)
        if pad_h_even:
            pad_h_left = pad_h_right = pad_h // 2
        else:
            pad_h_left = pad_h // 2
            pad_h_right = pad_h_left + 1
        if pad_w_even:
            pad_w_left = pad_w_right = pad_w // 2
        else:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w_left + 1
        image = np.pad(
            image,
            ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right), (0, 0)),
            "constant",
        )
        if self.transforms:
            for tfms in self.transforms:
                aug = tfms(
                    **{
                        "image": image,
                    }
                )
                image = aug["image"]
        return image

    def _random_crop(self, im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j

    def _get_pad(self, im_sz, crop_sz=512):
        h, w = im_sz
        h_mul = h // crop_sz + 1
        w_mul = w // crop_sz + 1
        pad_h = crop_sz * h_mul - h
        pad_w = crop_sz * w_mul - w
        return pad_h // 2, pad_w // 2, pad_h % 2 == 0, pad_w % 2 == 0


class Crop_Dataset_Video:
    def __init__(self, video_path, transforms=None, crop_size=512):
        self.frame = 0
        self.cap = cv2.VideoCapture(video_path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.crop_size = crop_size
        self.transforms = transforms

    def __iter__(self):
        return self

    def __next__(self):
        # Read video
        ret_val, img0 = self.cap.read()
        if not ret_val:  # end of video
            self.cap.release()
            raise StopIteration

        # Convert
        image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0
        h, w = image.shape[0], image.shape[1]
        pad_h, pad_w, pad_h_even, pad_w_even = self._get_pad((h, w), self.crop_size)
        if pad_h_even:
            pad_h_left = pad_h_right = pad_h // 2
        else:
            pad_h_left = pad_h // 2
            pad_h_right = pad_h_left + 1
        if pad_w_even:
            pad_w_left = pad_w_right = pad_w // 2
        else:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w_left + 1
        image = np.pad(
            image,
            ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right), (0, 0)),
            "constant",
        )
        if self.transforms:
            for tfms in self.transforms:
                aug = tfms(
                    **{
                        "image": image,
                    }
                )
                image = aug["image"]
        return image, img0

    def _get_pad(self, im_sz, crop_sz=512):
        h, w = im_sz
        h_mul = h // crop_sz + 1
        w_mul = w // crop_sz + 1
        pad_h = crop_sz * h_mul - h
        pad_w = crop_sz * w_mul - w
        return pad_h // 2, pad_w // 2, pad_h % 2 == 0, pad_w % 2 == 0
