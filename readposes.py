import numpy as np

npz_path = "/scratch/yz10442/must3r/output/1-A_d1uPKpnDksrjY3UE23dUTC0odvnHu/all_poses.npz"
data = np.load(npz_path, allow_pickle=True)

print("📂 Keys in file:", list(data.keys()))
for k in data.keys():
    v = data[k]
    if isinstance(v, np.ndarray):
        print(f"{k:15s} shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"{k:15s} type={type(v)}")
