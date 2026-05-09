import os, glob, torch
path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/classifier/version_64"
ckpt_dir = os.path.join(path, "checkpoints")
ckpt_file = max(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime)
ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
for k, v in ckpt['state_dict'].items():
    if 'classifier.head.0' in k:
        print(k, v.shape)
