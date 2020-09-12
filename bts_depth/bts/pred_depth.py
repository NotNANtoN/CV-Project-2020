from ycb_loader import PoseDataset
from pytorch.bts_modular import BTS

ds = PoseDataset(mode="test") # mode = train / test

img, depth, label, calibrate_params = ds[0]

print(img.shape, depth.shape, label, calibrate_params)

parser = None
depth_model = BTS(parser)
