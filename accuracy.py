import numpy as np

#Linear SVM
"""
tp = [392.0,377.0,372.0,370.0,354.0]
tn = [416.0,415.0,405.0,418.0,451.0]
fp = [224.0,226.0,253.0,242.0,238.0]
fn = [168.0,182.0,170.0,170.0,157.0]

for i in range(len(tp)):
	print tp[i]/(tp[i]+fp[i])
	print tp[i]/(tp[i]+fn[i])
	print (tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])
	print ''
"""
print 'Linear SVM'
pre = [63.636,62.52,59.52,60.457,59.792]
print np.mean(pre)
pre = [70.0,67.441,68.634,68.518,69.275]
print np.mean(pre)
pre = [67.334,68.0,64.75,65.667,66.083]
print np.mean(pre)
print np.sqrt(np.var(pre))
#Random forest
"""
tp = [391.0,373.0,370.0,373.0,371.0]
tn = [419.0,436.0,410.0,421.0,435.0]
fp = [225.0,230.0,255.0,239.0,221.0]
fn = [165.0,161.0,165.0,167.0,173.0]


for i in range(len(tp)):
	print tp[i]/(tp[i]+fp[i])
	print tp[i]/(tp[i]+fn[i])
	print (tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])
	print ''
"""
print 'Random Forest'
pre = [63.474,61.859,59.2,60.947,62.668]
print np.mean(pre)
pre = [70.323,69.85,69.158,69.074,68.198]
print np.mean(pre)
pre = [67.5,67.416,65.0,66.167,67.167]
print np.mean(pre)
print np.sqrt(np.var(pre))

#Gaussian
"""
tp = [179.0,179.0,160.0,177.0,182.0]
tn = [219.0,205.0,225.0,192.0,201.0]
fp = [136.0,121.0,162.0,129.0,127.0]
fn = [66.0,95.0,73.0,102.0,90.0]


for i in range(len(tp)):
	print tp[i]/(tp[i]+fp[i])
	print tp[i]/(tp[i]+fn[i])
	print (tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])
	print ''

"""

print 'Gaussian SVM'
pre = [56.825,59.662,49.689,57.843,58.899]
print np.mean(pre)
pre = [73.061,65.328,68.669,63.44,66.911]
print np.mean(pre)
pre = [66.334,64.0,62.096,61.5,63.834]
print np.mean(pre)
print np.sqrt(np.var(pre))

#Naive
"""
tp = [272.0,251.0,262.0,257.0,289.0]
tn = [90.0,95.0,81.0,90.0,66.0]
fp = [37.0,37.0,49.0,30.0,37.0]
fn = [201.0,217.0,208.0,223.0,208.0]


for i in range(len(tp)):
	print tp[i]/(tp[i]+fp[i])
	print tp[i]/(tp[i]+fn[i])
	print (tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])
	print ''
"""
print 'Naive'
pre = [88.025,87.152,84.244,89.547,88.65]
print np.mean(pre)
pre = [57.505,53.632,55.744,53.541,58.148]
print np.mean(pre)
pre = [60.334,57.667,57.167,57.833,59.166]
print np.mean(pre)
print np.sqrt(np.var(pre))

