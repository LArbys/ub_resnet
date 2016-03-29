import sys,os
import caffe
import numpy as np
import ROOT as rt
import lmdb
from math import log
from caffe.io import datum_to_array
from PIL import Image

gpu_id = 0

caffe.set_mode_gpu()
caffe.set_device(gpu_id)

# DEFINE DATA/MODEL FILES

# 768x768 padding
#test_data = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test.db"
#mean_file = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test_mean.bin"
test_data = "/mnt/disk0/taritree/larbys/prepared_lmdb/bnb_data_set1.db"
mean_file = "/mnt/disk0/taritree/larbys/prepared_lmdb/bnb_data_set1_mean.bin"
model = "training_attempts/v2/001/snapshot_rmsprop_iter_checkpointb.caffemodel"
deploy_prototxt  = "deploy_v2.prototxt"

prototxt = deploy_prototxt

# LOAD LMDB
lmdb_name = test_data
lmdb_env = lmdb.open(lmdb_name, readonly=True)
lmdb_txn = lmdb_env.begin()
cursor = lmdb_txn.cursor()

# CLASS DEFINITION
binlabels = {0:"background",1:"neutrino"}
classlabels = binlabels.keys()

# LOAD THE NET
net = caffe.Net( prototxt, model, caffe.TEST )
input_shape = net.blobs["data"].data.shape
images_copies_to_run = 2
if input_shape[0]%images_copies_to_run!=0:
    print "Images per Batch must be multiple of shape. %d/%d=%d"%(input_shape[0],images_copies_to_run,input_shape[0]%images_copies_to_run)
    sys.exit(-1)
print "We will process %d images per batch. Take ave. of %d images for the prob."%(input_shape[0]/images_copies_to_run,images_copies_to_run)

# MEAN IMAGE AND DATUM
# mean proto
fmean = open(mean_file,'rb')
mean_bin = fmean.read()
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(mean_bin)
mean_arr = np.array( caffe.io.blobproto_to_array(mean_blob) )
fmean.close()

data = np.zeros( input_shape, dtype=np.float32 )
input_labels = np.zeros( (input_shape[0],), dtype=np.float32 )
datum = caffe.proto.caffe_pb2.Datum()


print "[ENTER] to continue."
raw_input()


# LOOP OVER LMDB

ibatch = 0
nbatches = 5000
outofentries = False
totevents = 0

keys_to_run = ["05016_00036_0183900",
               "05013_00204_1022700",
               "05017_00020_0100300",
               "05016_00022_0112600"]
ikey = 0

while not outofentries:
    print "batch ",ibatch," of ",nbatches
    keys = []
    nfilled = 0

    # we do multiple crops for each image
    ngroups_this_batch = 0
    for group in range( input_shape[0]/images_copies_to_run ):
        foundit = True
        key = keys_to_run[ikey]
        raw_datum = lmdb_txn.get(b"%s"%(key))
        ikey += 1
        print key
        print "found: ",key
        ngroups_this_batch += 1
        datum.ParseFromString(raw_datum)
        vec = datum_to_array(datum)
        if key!="":
            keys.append(key)

        print "images_copies_to_run: ",images_copies_to_run
        for n in range(0,images_copies_to_run):
            if nfilled>=input_shape[0]:
                break
            # if only 1 image, center crop
            xoffset = int(0.5*(vec.shape[1]-input_shape[2]-1))
            yoffset = int(0.5*(vec.shape[2]-input_shape[3]-1))
            x1 = xoffset
            x2 = x1 + input_shape[2]
            y1 = yoffset
            y2 = y1 + input_shape[3]
            data[nfilled,:,:,:] = vec[:,x1:x2,y1:y2]-mean_arr[0,:,x1:x2,y1:y2]
            if nfilled==0:
                input_labels[nfilled] = 1 # always neutrino
            else:
                input_labels[nfilled] = 0
            nfilled += 1
            
            print "fill image"
            imgd = Image.fromarray( np.transpose( vec[:,x1:x2,y1:y2], (1,2,0) ), 'RGB' )
            imgd.save("image_%s.png"%(key))


        if ikey==len(keys_to_run):
            outofentries = True
            break


    net.set_input_arrays( data, input_labels )
    print np.max( data[0,:,:,:] )

    net.forward()
    net.backward()
    
    print "forward/backward"

    for group in range( ngroups_this_batch ):
        labels =  net.blobs["label"].data[group*images_copies_to_run:(group+1)*images_copies_to_run]
        scores = net.blobs["fc2"].data[group*images_copies_to_run:(group+1)*images_copies_to_run]
        probs = net.blobs["probt"].data[group*images_copies_to_run:(group+1)*images_copies_to_run]
        
        print group, labels[:,0,0,0],probs,scores
        key = keys[group]

        # Use mean
        labels = np.array( [np.mean(labels[:,0,0,0],axis=0)] )
        scores = np.mean(scores,axis=0)
        probs  = np.mean(probs,axis=0)
        decision = np.argmax(scores)
        most_nu = decision
        ilabel = int(labels[0])
        prob = probs
        score = scores

        # gradient?
        #print net.blobs["data"].diff
        grad_mag = np.fabs( net.blobs["data"].diff[0,:,:,:] )
        t_grad_mag = np.transpose( grad_mag, ( 1, 2, 0 ) )
        print "gradient: ",t_grad_mag.shape
        print np.max(t_grad_mag)
        t_grad_mag *= 255.0/np.max(t_grad_mag)

        img = Image.fromarray(t_grad_mag, 'RGB')
        img.save("saliency_nu_%s.png"%(key))

        grad_mag = np.fabs( net.blobs["data"].diff[1,:,:,:] )
        t_grad_mag = np.transpose( grad_mag, ( 1, 2, 0 ) )
        print "gradient: ",t_grad_mag.shape
        print np.max(t_grad_mag)
        t_grad_mag *= 255.0/np.max(t_grad_mag)

        img = Image.fromarray(t_grad_mag, 'RGB')
        img.save("saliency_cosmic_%s.png"%(key))



    ibatch += 1
    if ibatch>=nbatches:
        break
    #raw_input()
