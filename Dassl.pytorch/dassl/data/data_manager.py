import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image
from collections import defaultdict
from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform
import numpy as np
import pdb

def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0

    return data_loader


####################################################################################################################

# Federated DG setting

def build_data_loader_fed(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    num_users=30,
    alpha=0.5,
):
    if is_train == True:
        domain_dict = defaultdict(list)
        label_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            domain_dict[item.domain].append(i)
            label_dict[item.label].append(i)

        ########################################################################################
        # Multi-domain data distribution setup
        if cfg.DATALOADER.DATA_DISTRIBUTION == 'Multi':
            min_sample = 70
            min_size = 0

            while min_size < min_sample:
                dict_users = defaultdict(list)
                num_samples_per_user = [0 for _ in range(num_users)]
                max_samples_per_user = len(data_source) / num_users

                for c in range(n_domain):
                    idx_class_c = domain_dict[c]
                    np.random.shuffle(idx_class_c)

                    # dirichlet sampling from this label
                    proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                    proportions = np.array(
                        [p * (n_per_user < max_samples_per_user) for p, n_per_user in
                         zip(proportions, num_samples_per_user)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class_c)).astype(int)[:-1]

                    for user, idx_per_user in enumerate(np.split(idx_class_c, proportions)):
                        dict_users[user].extend(idx_per_user)
                        num_samples_per_user[user] = len(dict_users[user])

                min_size = min(num_samples_per_user)

        #########################################################################################
        # Single-domain data distribution setup
        elif cfg.DATALOADER.DATA_DISTRIBUTION == 'Single':
            num_device_same_domain = int(num_users / n_domain)
            dict_users = defaultdict(list)
            j = 0
            for c in range(n_domain):

                idx_class_c = domain_dict[c]
                np.random.shuffle(idx_class_c)
                partition = int(len(idx_class_c) // num_device_same_domain)


                for k in range(num_device_same_domain):
                    if k == num_device_same_domain -1:
                        dict_users[j].extend(idx_class_c[partition * k:])
                    else:
                        dict_users[j].extend(idx_class_c[partition*k:partition*(k+1)])
                    j += 1

        ###################################################################################################

        data_loaders = []
        for k in range(num_users):
            data_source_selected = [data_source[i] for i in dict_users[k]]

            sampler = build_sampler(
                sampler_type,
                cfg=cfg,
                data_source=data_source_selected,
                batch_size=batch_size,
                n_domain=n_domain,
                n_ins=n_ins,
            )

            if dataset_wrapper is None:
                dataset_wrapper = DatasetWrapper

            train_dataset = dataset_wrapper(cfg, data_source_selected, transform=tfm, is_train=is_train)

            # Build data loader
            data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=is_train and len(data_source) >= batch_size,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
            )
            assert len(data_loader) > 0

            data_loaders.append(data_loader)

        return data_loaders, dict_users

    else:

        # Build sampler
        sampler = build_sampler(
            sampler_type,
            cfg=cfg,
            data_source=data_source,
            batch_size=batch_size,
            n_domain=n_domain,
            n_ins=n_ins,
        )

        if dataset_wrapper is None:
            dataset_wrapper = DatasetWrapper

        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )
        assert len(data_loader) > 0

        return data_loader


def build_data_loader_class_balanced_fed(
    cfg,
    sampler_type="RandomClassSampler",
    data_source=None,
    batch_size=65,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    sample_idxs=None,
    num_users=30,
):
    data_loaders = []
    for k in range(num_users):
        data_source_t = [data_source[i] for i in sample_idxs[k]]

        # Build sampler
        sampler = build_sampler(
            "RandomClassSampler",
            cfg=cfg,
            data_source=data_source_t,
            batch_size=batch_size,
            n_domain=n_domain,
            n_ins=1,
        )

        if dataset_wrapper is None:
            dataset_wrapper = DatasetWrapper

        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source_t, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=is_train and len(data_source_t) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )

        assert len(data_loader) > 0
        data_loaders.append(data_loader)
    return data_loaders

class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        train_loader_x, dict_users = build_data_loader_fed(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=len(cfg.DATASET.SOURCE_DOMAINS),
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        train_loader_x_cb = build_data_loader_class_balanced_fed(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=dataset.num_classes,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
            sample_idxs=dict_users
        )

        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Class-balanced sampler
        self.train_loader_x_cb = train_loader_x_cb

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)


        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
