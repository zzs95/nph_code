import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.file_and_folder_operations import *
import monai
from monai.transforms import *
from copy import deepcopy
import torch.utils.data as data
import torch.nn as nn
def threshold_at(x):
    return x > 1
BATCHSIZE = 3
max_epoch = 100
lr = 1e-5
roi_size = (256, 256, 64)
result_dict = {}
# skull = ''
skull = '_noskull'
def main():
    modal = 'cross'
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    root_path = '/Projects/data_nph_new'
    flair_data_path = '/Projects/data_nph_new/brain_tumor_data'+skull+'_flair'
    t2_data_path = '/Projects/data_nph_new/brain_tumor_data'+skull+'_t2'
    label_df = pd.read_csv(os.path.join(root_path, 'cross_labels.csv'), dtype=str)

    images = label_df['mri_accession'].values.tolist()
    flair_modals = label_df['modal_name_x'].values.tolist()
    t2_modals = label_df['modal_name_y'].values.tolist()

    flair_image_list = [os.sep.join([flair_data_path, 'images', f + '-' + m.replace(' ', '_') + '.nii.gz']) for f, m in zip(images, flair_modals)]
    t2_image_list = [os.sep.join([t2_data_path, 'images', f + '-' + m.replace(' ', '_') + '.nii.gz']) for f, m in zip(images, t2_modals)]
    label_df = pd.merge(label_df, pd.DataFrame(flair_image_list, columns=['img_path_flair']), right_index=True, left_index=True)
    label_df = pd.merge(label_df, pd.DataFrame(t2_image_list, columns=['img_path_t2']), right_index=True, left_index=True)
    
    setnames = ['mrs_improve', 'cont_improve', 'gait_improve', 'cog_improvement']
    # setnames = ['cont_improve',]

    for setname in setnames:
        print('training ', setname)
        flair_checkpoint_dir = "./checkpoint_rih/flair"+skull+"/" + setname 
        t2_checkpoint_dir = "./checkpoint_rih/t2"+skull+"/" + setname 
        checkpoint_dir = "./checkpoint_rih/"+modal+skull+"/" + setname 
        maybe_mkdir_p(checkpoint_dir)
        log_dir = "./runs/"+modal+skull+"/" + setname 
        writer = SummaryWriter(log_dir=log_dir)
        set_df = deepcopy(label_df)
        set_df = set_df.dropna(axis=0, subset=[setname])
        images_flair = np.array(set_df['img_path_flair'])
        images_t2 = np.array(set_df['img_path_t2'])
        labels = set_df[setname].values.astype(np.int64)

        # random split
        # np.random.seed(177)
        # use mri_date split
        mri_years = np.array([a.split('-')[0] for a in set_df['mri_date'].values]).astype(int)
        train_idx = np.argwhere(mri_years < 2020).squeeze()
        test_idx = np.argwhere(mri_years >= 2020).squeeze()
        train_files = [{"img_flair": img1, "img_t2": img2, "label": label} for img1, img2, label in
                       zip(images_flair[train_idx], images_t2[train_idx], labels[train_idx])]
        test_files = [{"img_flair": img1, "img_t2": img2, "label": label} for img1, img2, label in
                       zip(images_flair[test_idx], images_t2[test_idx], labels[test_idx])]

        l_ts = 0
        for a in test_files:
            l_ts += a['label']
        l_tr = 0
        for a in train_files:
            l_tr += a['label']
        print('train file num:', len(train_files), 'positive num:', l_tr,
              'test file num:', len(test_files), 'positive num:', l_ts,
              )
        
        train_transforms = Compose(
            [
                LoadImaged(keys=["img_flair", "img_t2", ]),
                AddChanneld(keys=["img_flair", "img_t2"]),
                Orientationd(keys=["img_flair", "img_t2"], axcodes="RAS"),
                CropForegroundd(keys=["img_flair"], source_key="img_flair", select_fn=threshold_at),
                CropForegroundd(keys=[ "img_t2"], source_key="img_t2", select_fn=threshold_at),
                Resized(keys=["img_flair", "img_t2"], spatial_size=roi_size),
                ScaleIntensityRangePercentilesd(keys=["img_flair"], lower=0, upper=95, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityRangePercentilesd(keys=["img_t2"], lower=0, upper=95, b_min=0.0, b_max=1.0, clip=True),
                RandFlipd(
                    keys=["img_flair", "img_t2"],
                    spatial_axis=[0],
                    prob=0.50,
                ),
                RandZoomd(keys=["img_flair", "img_t2"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
                EnsureTyped(keys=["img_flair", "img_t2"]),
            ]
        )

        # create a training data loader
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

        # resample data
        target = torch.tensor([a['label'] for a in train_ds.data])
        class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
        mean_weight = 1. / class_sample_count.float()
        reverse_weight = mean_weight / class_sample_count.float()
        mean_samples_weight = torch.tensor([mean_weight[t] for t in target])
        reverse_samples_weight = torch.tensor([reverse_weight[t] for t in target])
        mean_sampler = data.WeightedRandomSampler(mean_samples_weight, len(mean_samples_weight))
        reverse_sampler = data.WeightedRandomSampler(reverse_samples_weight, len(reverse_samples_weight))

        train_loader = DataLoader(train_ds, batch_size=BATCHSIZE,
                                  # shuffle=True,
                                  num_workers=20,
                                  sampler=mean_sampler,
                                  pin_memory=torch.cuda.is_available())

        def load_pretrain(model, pth):
            from collections import OrderedDict
            pretrained_dict = torch.load(pth)
            new_state_dict = OrderedDict()

            for k, v in pretrained_dict.items():
                if not k in ['fc.weight', 'fc.bias']:  # remove `module.`
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            return model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class FuseNet(nn.Module):
            def __init__(self, spatial_dims=3, in_channels=1, out_channels=2):
                super(FuseNet, self).__init__()
                model_flair = monai.networks.nets.resnet50(spatial_dims=spatial_dims, n_input_channels=in_channels,
                                                           num_classes=out_channels, feed_forward=False).to(device)
                model_t2 = monai.networks.nets.resnet50(spatial_dims=spatial_dims, n_input_channels=in_channels,
                                                        num_classes=out_channels, feed_forward=False).to(device)
                load_pretrain(model_flair, os.path.join(flair_checkpoint_dir, "single_classification3d_dict.pth"))
                load_pretrain(model_t2, os.path.join(t2_checkpoint_dir, "single_classification3d_dict.pth"))
                self.feat1 = model_flair
                self.feat2 = model_t2

                self.class_layers = nn.Sequential(nn.Linear(4096, 2048),
                                                  nn.BatchNorm1d(2048),
                                                  nn.ReLU(True),
                                                  nn.Linear(2048, out_channels),
                                                  )

            def forward(self, x, x1):
                x = self.feat1(x)
                x1 = self.feat2(x1)
                x = self.class_layers(torch.cat([x, x1], dim=1))
                return x

        model = FuseNet(spatial_dims=3, in_channels=1, out_channels=2).to(device)

        loss_function = torch.nn.CrossEntropyLoss() # 2 class
        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5) # lr=1e-4
        

        # start a typical PyTorch training
        for epoch in range(max_epoch):
            print('.', end='')
            # print("-" * 10)
            # print(f"epoch {epoch + 1}/{max_epoch}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, inputs1, labels = batch_data["img_flair"].to(device), batch_data["img_t2"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs, inputs1)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            # print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.close()
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "fuse_classification3d_dict.pth"))


def main_single(modal):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_path = '/Projects/data_nph_new/brain_tumor_data'+skull+'_'+modal
    label_df = pd.read_csv(os.path.join(data_path, 'labels.csv'), dtype=str)

    images = label_df['mri_accession'].values.tolist()
    modals = label_df['modal_name'].values.tolist()

    image_list = [os.sep.join([data_path, 'images', f + '-' + m.replace(' ', '_') + '.nii.gz']) for f, m in zip(images, modals)]
    label_df = pd.merge(label_df, pd.DataFrame(image_list, columns=['img_path']), right_index=True, left_index=True)

    setnames = ['mrs_improve', 'cont_improve', 'gait_improve', 'cog_improvement']
    # setnames = ['cont_improve']
    for setname in setnames:
        print('training ', setname)
        checkpoint_dir = "./checkpoint_rih/"+modal+skull+"/" + setname 
        maybe_mkdir_p(checkpoint_dir)
        log_dir = "./runs/"+modal+skull+"/" + setname 
        writer = SummaryWriter(log_dir=log_dir)
        set_df = deepcopy(label_df)
        set_df = set_df.dropna(axis=0, subset=[setname])
        images = np.array(set_df['img_path'])
        labels = set_df[setname].values.astype(np.int64)

        # random split
        np.random.seed(177)
        # use mri_date split
        mri_years = np.array([a.split('-')[0] for a in set_df['mri_date'].values]).astype(int)
        train_idx = np.argwhere(mri_years < 2020).squeeze()
        test_idx = np.argwhere(mri_years >= 2020).squeeze()
        train_files = [{"img": img, "label": label} for img, label in zip(images[train_idx], labels[train_idx])]
        test_files = [{"img": img, "label": label} for img, label in zip(images[test_idx], labels[test_idx])]

        l_ts = 0
        for a in test_files:
            l_ts += a['label']
        l_tr = 0
        for a in train_files:
            l_tr += a['label']
        print('train file num:', len(train_files), 'positive num:', l_tr,
              'test file num:', len(test_files), 'positive num:', l_ts,
              )
        train_transforms = Compose(
            [
                LoadImaged(keys=["img"]),
                AddChanneld(keys=["img"]),
                Orientationd(keys=["img", ], axcodes="RAS"),
                CropForegroundd(keys=["img"], source_key="img", select_fn=threshold_at),
                Resized(keys=["img"], spatial_size=roi_size),
                ScaleIntensityRangePercentilesd(keys=["img"], lower=0, upper=95, b_min=0.0, b_max=1.0, clip=True),
                RandFlipd(
                    keys=["img",],
                    spatial_axis=[0],
                    prob=0.50,
                ),
                RandZoomd(keys=["img",], min_zoom=0.9, max_zoom=1.1, prob=0.5),
                EnsureTyped(keys=["img"]),
            ]
        )

        # create a training data loader
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

        # resample data
        target = torch.tensor([a['label'] for a in train_ds.data])
        class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
        mean_weight = 1. / class_sample_count.float()
        reverse_weight = mean_weight / class_sample_count.float()
        mean_samples_weight = torch.tensor([mean_weight[t] for t in target])
        reverse_samples_weight = torch.tensor([reverse_weight[t] for t in target])
        mean_sampler = data.WeightedRandomSampler(mean_samples_weight, len(mean_samples_weight))
        reverse_sampler = data.WeightedRandomSampler(reverse_samples_weight, len(reverse_samples_weight))

        train_loader = DataLoader(train_ds, batch_size=BATCHSIZE,
                                  # shuffle=True,
                                  num_workers=20,
                                  sampler=mean_sampler,
                                  pin_memory=torch.cuda.is_available())

        # Create resnet50, CrossEntropyLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)

        loss_function = torch.nn.CrossEntropyLoss() # 2 class
        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)

        # start a typical PyTorch training
        max_epoch = 60
        for epoch in range(max_epoch):
            print('.', end='')
            # print("-" * 10)
            # print(f"epoch {epoch + 1}/{max_epoch}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = nn.functional.softmax(outputs, dim=1)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            # print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        writer.close()
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "single_classification3d_dict.pth"))

if __name__ == "__main__":
    modals = ['flair', 't2']
    for modal in modals:
        main_single(modal )
    main()
