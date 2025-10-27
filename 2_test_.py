import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from Utils.file_and_folder_operations import *
import monai
from monai.metrics import ROCAUCMetric
from monai.transforms import *
from copy import deepcopy
import torch.nn as nn
def threshold_at(x):
    # threshold at 1
    return x > 1
BATCHSIZE = 3
roi_size = (256, 256, 64)
result_dict = {}
# skull = ''
skull = '_noskull'
result_dict = {}
def test_single(modal):
    root_path = '/Projects/data_nph_new/'
    test_dict_path = join(root_path, 'test_rih'+skull+'_dict')
    maybe_mkdir_p(test_dict_path)
    data_path = join(root_path, 'brain_tumor_data'+skull+'_'+modal)
    label_df = pd.read_csv(os.path.join(data_path, 'labels.csv'), dtype=str)

    images = label_df['mri_accession'].values.tolist()
    modals = label_df['modal_name'].values.tolist()

    image_list = [os.sep.join([data_path, 'images', f + '-' + m.replace(' ', '_') + '.nii.gz']) for f, m in zip(images, modals)]
    label_df = pd.merge(label_df, pd.DataFrame(image_list, columns=['img_path']), right_index=True, left_index=True)

    setnames = ['mrs_improve', 'cont_improve', 'gait_improve', 'cog_improvement']
    # setnames = ['cont_improve']
    set_out_accs = []
    set_out_aucs = []
    for setname in setnames:
        curr_df = label_df[['mri_accession', 'mri_date']+[setname]+['img_path']]
        # print('testing ', setname)
        checkpoint_dir = "./checkpoint_rih/"+modal+skull+"/" + setname 
        set_df = deepcopy(curr_df)
        set_df = set_df.dropna(axis=0, subset=[setname])
        images = np.array(set_df['img_path'])
        labels = set_df[setname].values.astype(np.int64)

        # use mri_date split
        mri_years = np.array([a.split('-')[0] for a in set_df['mri_date'].values]).astype(int)
        test_idx = np.argwhere(mri_years >= 2020).squeeze()
        test_files = [{"img": img, "label": label} for img, label in zip(images[test_idx], labels[test_idx])]
        test_df = copy.deepcopy(set_df.iloc[test_idx])

        val_transforms = Compose(
            [
                LoadImaged(keys=["img"]),
                AddChanneld(keys=["img"]),
                Orientationd(keys=["img", ], axcodes="RAS"),
                CropForegroundd(keys=["img"], source_key="img", select_fn=threshold_at),
                Resized(keys=["img"], spatial_size=roi_size),
                # ScaleIntensityRanged(keys=["img"], a_min=0, a_max=1024, b_min=0.0, b_max=1.0, clip=False),
                ScaleIntensityRangePercentilesd(keys=["img"], lower=0, upper=95, b_min=0.0, b_max=1.0, clip=True),
                EnsureTyped(keys=["img"]),
            ]
        )
        post_pred = Compose([EnsureType(), Activations(softmax=True)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        # post_pred = Compose([EnsureType(), Activations(sigmoid=True)])

        # create a validation data loader
        test_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, num_workers=20,
                                pin_memory=torch.cuda.is_available())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "single_classification3d_dict.pth")))
        auc_metric = ROCAUCMetric()

        # start a typical PyTorch training
        best_metric = {}
        best_metric['acc'] = -1
        best_metric['auc'] = -1

        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for test_data in test_loader:
                val_images, val_labels = test_data["img"].to(device), test_data["label"].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)

            y_pred_argmax = y_pred.argmax(dim=1)
            acc_value = torch.eq(y_pred_argmax, y) # 2 class
            acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i).cpu() for i in y]  # 2 class
            y_pred_act = [post_pred(i).cpu() for i in y_pred]  # 2 class
            auc_metric(y_pred_act, y_onehot) # 2 class
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            best_metric['acc'] = acc_metric
            best_metric['auc'] = auc_result
        y_prob = torch.concat(y_pred_act).reshape(-1, 2)[:,1]
        y_prob = y_prob.cpu().numpy().tolist()
        test_df.insert(loc=4, column='pred_prob', value=y_prob)
        test_df.insert(loc=5, column='pred_clas', value=y_pred_argmax.cpu().numpy().tolist())
        test_xlsx = join(test_dict_path,  modal+'-'+setname+'.xlsx')
        test_df.to_excel(test_xlsx)

        set_out_accs.append(f"{best_metric['acc']:.4f}")
        set_out_aucs.append(f"{best_metric['auc']:.4f}")
    for i in range(len(setnames)):
        result_dict[modal+'_'+setnames[i]] = {}
        print(setnames[i], 'acc:', set_out_accs[i], 'auc:', set_out_aucs[i])
        result_dict[modal+'_'+setnames[i]]['acc'] = set_out_accs[i]
        result_dict[modal+'_'+setnames[i]]['auc'] = set_out_aucs[i]
        
def test_cross():
    modal = 'cross'
    root_path = '/Projects/data_nph_new'
    flair_data_path = '/Projects/data_nph_new/brain_tumor_data'+skull+'_flair'
    t2_data_path = '/Projects/data_nph_new/brain_tumor_data'+skull+'_t2'
    test_dict_path = join(root_path, 'test_rih'+skull+'_dict')
    maybe_mkdir_p(test_dict_path)

    label_df = pd.read_csv(os.path.join(root_path, 'cross_labels.csv'), dtype=str)
    images = label_df['mri_accession'].values.tolist()
    flair_modals = label_df['modal_name_x'].values.tolist()
    t2_modals = label_df['modal_name_y'].values.tolist()

    flair_image_list = [os.sep.join([flair_data_path, 'images', f + '-' + m.replace(' ', '_') + '.nii.gz']) for f, m in zip(images, flair_modals)]
    t2_image_list = [os.sep.join([t2_data_path, 'images', f + '-' + m.replace(' ', '_') + '.nii.gz']) for f, m in zip(images, t2_modals)]
    label_df = pd.merge(label_df, pd.DataFrame(flair_image_list, columns=['img_path_flair']), right_index=True, left_index=True)
    label_df = pd.merge(label_df, pd.DataFrame(t2_image_list, columns=['img_path_t2']), right_index=True, left_index=True)

    setnames = ['mrs_improve', 'cont_improve', 'gait_improve', 'cog_improvement']
    # setnames = ['cont_improve']

    set_out_accs = []
    set_out_aucs = []
    for setname in setnames:
        # print('testing ', setname)
        curr_df = label_df[['mri_accession', 'mri_date']+[setname]+['img_path_flair', 'img_path_t2']]
        checkpoint_dir = "./checkpoint_rih/"+modal+skull+"/" + setname 
        set_df = deepcopy(curr_df)
        set_df = set_df.dropna(axis=0, subset=[setname])
        test_df = copy.deepcopy(set_df)
        images_flair = np.array(set_df['img_path_flair'])
        images_t2 = np.array(set_df['img_path_t2'])
        labels = set_df[setname].values.astype(np.int64)

        # use mri_date split
        mri_years = np.array([a.split('-')[0] for a in set_df['mri_date'].values]).astype(int)
        test_idx = np.argwhere(mri_years >= 2020).squeeze()
        test_files = [{"img_flair": img1, "img_t2": img2, "label": label} for img1, img2, label in
                       zip(images_flair[test_idx], images_t2[test_idx], labels[test_idx])]
        test_df = copy.deepcopy(set_df.iloc[test_idx])
        val_transforms = Compose(
            [
                LoadImaged(keys=["img_flair", "img_t2"]),
                AddChanneld(keys=["img_flair", "img_t2"]),
                Orientationd(keys=["img_flair", "img_t2"], axcodes="RAS"),
                CropForegroundd(keys=["img_flair"], source_key="img_flair", select_fn=threshold_at),
                CropForegroundd(keys=[ "img_t2"], source_key="img_t2", select_fn=threshold_at),
                Resized(keys=["img_flair", "img_t2"], spatial_size=roi_size),
                ScaleIntensityRangePercentilesd(keys=["img_flair"], lower=0, upper=95, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityRangePercentilesd(keys=["img_t2"], lower=0, upper=95, b_min=0.0, b_max=1.0, clip=True),
                EnsureTyped(keys=["img_flair", "img_t2"]),
            ]
        )
        post_pred = Compose([EnsureType(), Activations(softmax=True)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

        # create a validation data loader
        test_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, num_workers=20,
                                pin_memory=torch.cuda.is_available())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class FuseNet(nn.Module):
            def __init__(self, spatial_dims=3, in_channels=1, out_channels=2):
                super(FuseNet, self).__init__()

                model_flair = monai.networks.nets.resnet50(spatial_dims=spatial_dims, n_input_channels=in_channels,
                                                           num_classes=out_channels, feed_forward=False).to(device)
                model_t2 = monai.networks.nets.resnet50(spatial_dims=spatial_dims, n_input_channels=in_channels,
                                                        num_classes=out_channels, feed_forward=False).to(device)
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
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "fuse_model_classification3d_dict.pth")))
        auc_metric = ROCAUCMetric()
        # start a typical PyTorch training
        best_metric = {}
        best_metric['acc'] = -1
        best_metric['auc'] = -1

        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for test_data in test_loader:
                test_images, test_images1, val_labels = test_data["img_flair"].to(device), test_data["img_t2"].to(device), test_data["label"].to(device)
                y_pred = torch.cat([y_pred, model(test_images, test_images1)], dim=0)
                y = torch.cat([y, val_labels], dim=0)

            y_pred_argmax = y_pred.argmax(dim=1)
            acc_value = torch.eq(y_pred_argmax, y) # 2 class
            acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i).cpu() for i in y]  # 2 class
            y_pred_act = [post_pred(i).cpu() for i in y_pred]  # 2 class
            auc_metric(y_pred_act, y_onehot) # 2 class
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            best_metric['acc'] = acc_metric
            best_metric['auc'] = auc_result
            # print(
            #     "testing cross, best accuracy: {:.4f} best AUC: {:.4f}".format(
            #         acc_metric, auc_result,
            #     )
            # )
        set_out_accs.append(f"{best_metric['acc']:.4f}")
        set_out_aucs.append(f"{best_metric['auc']:.4f}")

        y_prob = torch.concat(y_pred_act).reshape(-1, 2)[:,1]
        y_prob = y_prob.cpu().numpy().tolist()
        test_df.insert(loc=4, column='pred_prob', value=y_prob)
        test_df.insert(loc=5, column='pred_clas', value=y_pred_argmax.cpu().numpy().tolist())
        test_xlsx = join(test_dict_path,  modal+'-'+setname+'.xlsx')
        test_df.to_excel(test_xlsx)
    for i in range(len(setnames)):
        result_dict[modal+'_'+setnames[i]] = {}
        print(setnames[i], 'acc:', set_out_accs[i], 'auc:', set_out_aucs[i])
        result_dict[modal+'_'+setnames[i]]['acc'] = set_out_accs[i]
        result_dict[modal+'_'+setnames[i]]['auc'] = set_out_aucs[i]


if __name__ == "__main__":
    modals = ['flair', 't2']
    for modal in modals:
        test_single(modal)
    test_cross()
    df = pd.DataFrame.from_dict(result_dict).transpose()
    df.to_excel('test_rih_dict'+skull+'.xlsx')
            