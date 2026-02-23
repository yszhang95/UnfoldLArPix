#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

threshold = 4
filtered_deconv_q = []
filtered_smeared_true = []
filtered_totQ = []
for ievent in range(1):
  f = np.load('deconv_event_0_0.npz')
  smeared_true = f['smeared_true']
  deconv_q = f['deconv_q'] * (f['deconv_q'] > threshold)
  effq_proj = np.sum(smeared_true, axis=-1)
  deconv_proj = np.sum(deconv_q, axis=-1)
  print(np.sum(deconv_proj), deconv_proj[deconv_proj > 0].shape)
  print((deconv_proj > 0).shape, deconv_proj.shape, deconv_q.shape, 'check', (deconv_q > 3.5).shape, np.sum(deconv_q > 3.5),
        )

  # start hits
  hl = f['hits_location']
  hd = f['hits_data']
  totQ = {}
  for i, (l, d) in enumerate(zip(hl, hd)):
      q = totQ.get((l[0], l[1]), 0)
      q += d[-1]
      totQ[(l[0], l[1])] = q

  totQ = np.array(list(totQ.values()))

  filtered_deconv_q.extend(deconv_proj[deconv_proj > threshold].tolist())
  filtered_smeared_true.extend(effq_proj[effq_proj > threshold].tolist())
  filtered_totQ.extend(totQ[totQ > 1].tolist())


plt.figure(figsize=(10, 6))
plt.hist(filtered_smeared_true, label='Smeared True Projection', range=(0, 40), bins=40, alpha=0.5)
plt.hist(filtered_deconv_q, label='Deconvolved Projection', range=(0, 40), bins=40, alpha=0.5)
plt.hist(filtered_totQ, label='Total Charge from Hits', range=(0, 40), bins=40, alpha=0.5)
plt.xlabel('Charge Projection')
plt.ylabel('Count')
plt.title('Comparison of Smeared True Projection and Deconvolved Projection')
plt.legend()
plt.grid()
plt.savefig('hist_deconv_q.png')
