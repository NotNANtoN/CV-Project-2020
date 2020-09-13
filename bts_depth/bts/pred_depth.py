import os

import torch
import numpy as np
from tqdm import tqdm

from ycb_loader import PoseDataset
from pytorch.bts_modular import BTS


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
parser = None
depth_model = BTS(parser)
depth_model.cuda()
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

targets = []
preds = []
names = []

root = '/data_b/8wiehe/depth_preds'

save = True

count = 0 
for img, depth, label, calibrate_params, folder in tqdm(dl):
    #img = undistort_image(img,)
    img = torch.tensor(img).float().cuda()
    depth = torch.tensor(depth)
    pred_depth = depth_model.forward(img)
    
    if save:
        for pred, target, name in zip(pred_depth, depth, folder):
            path = os.path.join(root, name)
            os.makedirs(path, exist_ok=True)
            torch.save(pred, path + "depth_pred.pt")
            names.append(name)

    else:
        targets.append(depth.float().cpu())
        preds.append(pred_depth.cpu())
        names.append(folder)

        count += 1
        #if count == 60:
        #    break
torch.save(names, os.path.join(root, 'all_paths.pt'))

if save:
    quit()

print("Done")
targets = torch.cat(targets).squeeze() / 100 / 100
preds = torch.cat(preds).squeeze() / 10

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
