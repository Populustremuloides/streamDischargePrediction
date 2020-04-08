import matplotlib.pyplot as plt
import numpy as np

burnIn = 30

sf = np.load("val_ref_sourceFlow.npy", allow_pickle=True)
tf = np.load("val_ref_targetFlow.npy", allow_pickle=True)
pf = np.load("val_ref_preditedFlow.npy", allow_pickle=True)

print(sf)
print(tf)
print(pf)

plt.plot(tf, label="targetFlow")
plt.plot(sf, label="sourceFlow")
plt.plot(pf, label="predictedFlow")

plt.legend()
plt.show()
