import sys,os
import caffe
import numpy as np
import lmdb
from math import log
from caffe.io import datum_to_array

caffe.set_mode_cpu()

mean_file = sys.argv[1]

# mean proto
fmean = open(mean_file,'rb')
mean_bin = fmean.read()
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(mean_bin)
mean_arr = np.array( caffe.io.blobproto_to_array(mean_blob) )
fmean.close()
#(1, 3, 788, 768)
print mean_arr.shape

cropped = mean_arr[:,:,10:778,:]

print cropped.shape

out_arr = np.zeros((768,768),dtype=np.float)
for i in range(0,3):
    out_arr[:,:] += cropped[0,i,:,:]
out_arr /= 3.0

print out_arr.shape

out2 = mean_arr[0,:,:,:]

print out2.shape

fout = open('mean.npy','w')
np.save(fout,out2)
#np.save(fout,out_arr)
fout.close()
