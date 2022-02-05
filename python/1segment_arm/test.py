import numpy as np


deformations = [0.66904594, 1.856359429, 4.974174463, 6.834491314, 9.996107754, 13.598739041, 22.019953669, 32.449436599]
deformations = [-1.5973748590351056,  -2.8267911743902925, -5.907451873400103, -7.803911136259951, -10.964765077879058, -14.673825290684249, -22.982862859057516, -33.466075380070585]
diff = [0.39562683273905996, 1.4575720278240387, 0.38519845402592345, 0.7030851136775649, 0.14904992458333233, 0.8855394469951978, 0.7231276397639661, 0.7140968170765092]

rel = []
for d, tot in zip(diff, deformations):
    rel.append(-d/tot*100)
    
np.set_printoptions(precision=2)
print(np.array(rel))
print(np.array(rel).mean())
