import os
import glob
from pathlib import Path
from torch.utils import data
from torchvision.transforms import Normalize
import torch
import numpy as np
import argparse


class StarDataSet(data.Dataset):
    """
    Dataset personalizado de PyTorch.
    Debe implementar __init__, __len__ y __getitem__.
    """

    def __init__(
        self,
        split="train",
        data_dir=None,
        norm=False,
        mean=None,
        std=None,
        random_crop=False,
        patch_size=None,
        pixel_size=6/1000.0,
    ):
        assert (split in ["train", "val", "test"])

        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[1] / "data_generation" / "training_data"

        self.data_dir = Path(data_dir)

        self.img_dir = self.data_dir / (split + "_raw")
        self.dist_map_dir = self.data_dir / (split + "_dist_map")
        self.seg_map_dir = self.data_dir / (split + "_seg_map")
        self.centroid_dir = self.data_dir / (split + "_centroid")

        self.filenames = [
            os.path.splitext(os.path.basename(l))[0] for l in glob.glob(str(self.img_dir / "*.npy"))
        ]

        self.norm = norm
        self.mean = mean
        self.std = std
        self.random_crop = random_crop
        self.pixel_size = pixel_size

        if patch_size is None:
            self.patch_size = None
        elif isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = tuple(patch_size)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        img = np.load(self.img_dir / (filename + ".npy")).astype(np.float32)

        dist_map = np.load(self.dist_map_dir / (filename.replace("raw_image", "dist_map") + ".npy")).astype(np.float32)
        seg_map = np.load(self.seg_map_dir / (filename.replace("raw_image", "seg_map") + ".npy")).astype(np.float32)
        centroid = np.load(self.centroid_dir / (filename.replace("raw_image", "centroid") + ".npy")).astype(np.float32)

        if centroid.ndim == 1:
            centroid = centroid.reshape(
                1, -1) if centroid.size > 0 else np.zeros((0, 4), dtype=np.float32)

        if self.random_crop and self.patch_size is not None:
            patch_h, patch_w = self.patch_size
            img_h, img_w = img.shape

            if img_h > patch_h and img_w > patch_w:
                offset_y = np.random.randint(0, img_h - patch_h + 1)
                offset_x = np.random.randint(0, img_w - patch_w + 1)

                img = img[offset_y:offset_y + patch_h,
                          offset_x:offset_x + patch_w]
                dist_map = dist_map[offset_y:offset_y +
                                    patch_h, offset_x:offset_x + patch_w]
                seg_map = seg_map[offset_y:offset_y +
                                  patch_h, offset_x:offset_x + patch_w]

                if centroid.shape[0] > 0:
                    min_u = offset_x * self.pixel_size
                    max_u = (offset_x + patch_w) * self.pixel_size
                    min_v = offset_y * self.pixel_size
                    max_v = (offset_y + patch_h) * self.pixel_size

                    in_bbox = (
                        (centroid[:, 0] > min_u)
                        & (centroid[:, 0] < max_u)
                        & (centroid[:, 1] > min_v)
                        & (centroid[:, 1] < max_v)
                    )
                    centroid = centroid[in_bbox].copy()

                    if centroid.shape[0] > 0:
                        centroid[:, 0] = centroid[:, 0] - \
                            (offset_x * self.pixel_size)
                        centroid[:, 1] = centroid[:, 1] - \
                            (offset_y * self.pixel_size)
                else:
                    centroid = np.zeros((0, 4), dtype=np.float32)

        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)  # agrega un canal falso

        dist_map = torch.tensor(dist_map, dtype=torch.float32)
        # dist_map = dist_map.unsqueeze(0)  # agrega un canal falso

        seg_map = torch.tensor(seg_map, dtype=torch.float32)
        # seg_map = seg_map.unsqueeze(0)  # agrega un canal falso

        centroid = torch.tensor(centroid, dtype=torch.float32)

        if self.norm:
            # Normaliza por canal con mean y std.
            # El set de test usa los mismos valores del train.
            img = Normalize(self.mean, self.std)(img)

        # Asegura memoria contigua.
        return img.contiguous(), dist_map, seg_map, centroid


def star_collate_fn(batch):
    """Mantiene centroides de largo variable como lista."""
    imgs, dist_maps, seg_maps, centroids = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    dist_maps = torch.stack(dist_maps, dim=0)
    seg_maps = torch.stack(seg_maps, dim=0)
    return imgs, dist_maps, seg_maps, list(centroids)


def compute_mean_std(train_dataloader):
    first_batch = next(iter(train_dataloader))
    num_channel = first_batch[0].size(1)

    total_pixels = 0
    mean = np.zeros(num_channel, dtype=np.float64)

    for batch in train_dataloader:
        imgs = batch[0]
        b, c, h, w = imgs.shape
        total_pixels += b * h * w
        mean += imgs.sum(dim=(0, 2, 3)).cpu().numpy()

    mean = mean / total_pixels

    std = np.zeros(num_channel, dtype=np.float64)
    for batch in train_dataloader:
        imgs = batch[0]
        centered = imgs - \
            torch.tensor(mean, dtype=imgs.dtype).view(1, -1, 1, 1)
        std += (centered.pow(2).sum(dim=(0, 2, 3))).cpu().numpy()
    std = np.sqrt(std / total_pixels)

    return mean, std


if __name__ == '__main__':
    # python .\data_load.py --cal_mean_std 0 --vis 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_mean_std", type=int, default=1)
    parser.add_argument("--vis", type=int, default=1)
    args = parser.parse_args()

    # Calcula mean y std del set de entrenamiento.
    if args.cal_mean_std == 1:
        train_dataset = StarDataSet(split='train')
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )
        mean, std = compute_mean_std(train_dataloader)
    else:
        mean = [44.1619381]
        std = [60.98225565]
    print("means of training set are {}".format(mean))
    print("standard deviations of training set are {}".format(std))

    # Normaliza datos.
    train_dataset = StarDataSet(split='train', norm=True, mean=mean, std=std)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Carga val y test con la misma normalización del train.
    val_dataset = StarDataSet(split='val', norm=True, mean=mean, std=std)
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    test_dataset = StarDataSet(split='test', norm=True, mean=mean, std=std)
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    batch_val = next(iter(val_dataloader))
    batch_test = next(iter(test_dataloader))
    print("the dimension of a single raw images batch of validation set is {}".format(batch_val[0].size()))
    print("the dimension of a single dist map batch of test set is {}".format(batch_test[1].size()))
    print("the dimension of a single seg map batch of test set is {}".format(batch_test[2].size()))

    # Obtiene un batch del dataloader.
    # iter() crea el iterador.
    # next() devuelve el siguiente elemento.
    batch = next(iter(train_dataloader))
    images, dist_map, seg_map, centroid_real = batch
    print("the dimension of a single centroid batch of training set is {}".format(batch[3].size()))
