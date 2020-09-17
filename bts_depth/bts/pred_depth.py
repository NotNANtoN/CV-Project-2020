import os
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from ycb_loader import PoseDataset
from pytorch.bts_modular import define_parser, read_args, BTS



def undistort_image(img, camera_matrix, dist_coefs):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst

ds = PoseDataset(mode="test") # mode = train / test

img, depth, label, calibrate_params, folder = ds[0]

print(len(ds), img.shape, depth.shape, label.shape, calibrate_params, folder)


# args: checkpoint_path pytorch/models/bts_nyu_v2_pytorch_densenet121/model
parser = define_parser()
#parser = extend_parser(parser)
args = read_args(parser)
args.checkpoint_path = "pytorch/models/bts_nyu_v2_pytorch_densenet121/model"
depth_model = BTS(None, args=args)
#weights = torch.load("finetuned2020-09-16 16:09:49.pt")
#depth_model.load_state_dict(weights)
depth_model.cuda()
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

targets = []
preds = []
names = []

root = '/data_b/8wiehe/depth_preds'

save = False

rmse = 0
mae = 0
abs_rel = 0
rmse_dict = defaultdict(int)
mae_dict = defaultdict(int)
abs_rel_dict = defaultdict(int)
count_dict = defaultdict(int)

def sum_mean(x):
    return x.mean(-1).mean(-1).sum()

count = 0 
for img, depth, label, calibrate_params, folder in tqdm(dl):
    #img = undistort_image(img,)
    img = torch.tensor(img).float().cuda()
    depth = torch.tensor(depth).float() / 10000
    with torch.no_grad():
        pred_depth = depth_model.forward(img).cpu().squeeze().float() / 10
        
    if save:
        for pred, target, name in zip(pred_depth, depth, folder):
            path = os.path.join(root, name)
            os.makedirs(path, exist_ok=True)
            torch.save(pred, path + "depth_pred.pt")
            names.append(name)

    else:
        diff = (depth - pred_depth)
        abs_ = diff.abs()
        
        rmse += sum_mean(diff.pow(2))
        mae += sum_mean(abs_)
        abs_rel += sum_mean(abs_ / (depth + 1))
        count += len(depth)
        
        for class_num in label.unique():
            class_num = class_num.item()
            mask = label == class_num
            
            rmse_dict[class_num] += sum_mean(diff[mask].pow(2))
            mae_dict[class_num] += sum_mean(abs_[mask])
            abs_rel_dict[class_num] += sum_mean(abs_[mask] / (depth[mask] + 1))
            # count num images of batch that the class_num object appeared in :
            class_count = (mask.squeeze().sum(-1).sum(-1) > 0).sum()
            count_dict[class_num] += class_count
        
        #targets.append(depth.float().cpu())
        #preds.append(pred_depth.cpu())
        #names.append(folder)

        #count += 1
        #if count == 60:
        #    break

if save:
    torch.save(names, os.path.join(root, 'all_paths.pt'))
    quit()
    
    
rmse = (rmse / count).sqrt().item()
mae = (mae / count).item()
abs_rel = (abs_rel / count).item()
for class_num in count_dict:
    class_count = count_dict[class_num]
    rmse_dict[class_num] = (rmse_dict[class_num] / class_count).item()
    mae_dict[class_num] = (mae_dict[class_num] / class_count).item()
    abs_rel_dict[class_num] = (abs_rel_dict[class_num] / class_count).item()
def print_dict(dict_):
    for key in dict_:
        print(key, dict_[key])
print("RMSE classes: ")
print_dict(rmse_dict)
print("MAE classes: ")
print_dict(mae_dict)
print("Abs Rel MAE classes: ")
print_dict(abs_rel_dict)
print("TABLE: ")
print("Class  RMSE  MAE  Abs.MAE")
for key in rmse_dict:
    print(f'{key} & {rmse_dict[key]:.3f} & {mae_dict[key]:.3f} & {abs_rel_dict[key]:.3f}')
print("RMSE: ", rmse)
print("MAE: ", mae)
print("Abs Rel MAE: ", abs_rel)
print("Count: ", count)
print("Mean RMSE classes: ", sum(list(rmse_dict.values())) / len(rmse_dict))
print("Mean MAE classes: ", sum(list(mae_dict.values())) / len(mae_dict))
print("Mean Abs rel MAE classes: ", sum(list(abs_rel_dict.values())) / len(abs_rel_dict))

quit()

print("Done")
targets = torch.cat(targets).squeeze()
preds = torch.cat(preds).squeeze()

torch.save(targets, "targets.pt")
torch.save(preds, "preds.pt")
torch.save(names, "paths.pt")

from torchvision.utils import save_image
save_image(preds[:9].unsqueeze(1) / preds.max(), "preds.png", nrow=3)
save_image(targets[:9].unsqueeze(1) / targets.max(), "targets.png", nrow=3)



print("Shapes: ", targets.shape, preds.shape)
print("max, min, mean of preds: ", preds.max(), preds.min(), preds.mean())
print("max, min, mean of targets: ", targets.max(), targets.min(), targets.mean())


normed_preds = preds / preds.max() 
normed_preds *= targets.max()
normed_targets = targets# / targets.abs().max()
normed_rmse = (normed_targets - normed_preds).pow(2).mean().sqrt()
        
rmse = (targets - preds).pow(2).mean().sqrt()

abs_rel = ((targets - preds).abs() / (targets + 0.001)).mean()
print("RMSE: ", rmse)
print("Abs rel: ", abs_rel)
print("Normed RMSE: ", normed_rmse)
