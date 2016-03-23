import sys,os
import caffe
import numpy as np
import ROOT as rt
import lmdb
from math import log
from caffe.io import datum_to_array

gpu_id = 0

caffe.set_mode_gpu()
caffe.set_device(gpu_id)

test_prototxt = "training_attempts/0017/test.prototxt"
data_prototxt = "training_attempts/0017/data.prototxt"
deploy_prototxt  = "training_attempts/0017/deploy.prototxt"
model = "training_attempts/0017/snapshot_rmsprop_iter_50000.caffemodel"
train_data = "/home/taritree/working/larbys/staged_data/resized_traindata_combinedbnbcosmics.db"

#test_data  = "/home/taritree/working/larbys/staged_data/resized_testdata_combinedbnbcosmics.db"
#mean_file = "/home/taritree/working/larbys/staged_data/resized_testdata_combinedbnbcosmics_mean.bin"

#test_data = "/home/taritree/working/larbys/staged_data/resized_databnb.db"
#mean_file = "/home/taritree/working/larbys/staged_data/resized_databnb_mean.bin"

#test_data = "/home/taritree/working/larbys/staged_data/resized_databnb_set2.db"
#mean_file = "/home/taritree/working/larbys/staged_data/resized_databnb_set2_mean.bin"

#test_data = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test.db"
#mean_file = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test_mean.bin"
test_data = "/mnt/disk0/taritree/larbys/prepared_lmdb/bnb_data_set1.db"
mean_file = "/mnt/disk0/taritree/larbys/prepared_lmdb/bnb_data_set1_mean.bin"
model = "training_attempts/v2/001/snapshot_rmsprop_iter_checkpointb.caffemodel"
deploy_prototxt  = "deploy_v2.prototxt"

prototxt = deploy_prototxt
net = caffe.Net( prototxt, model, caffe.TEST )

lmdb_name = test_data
lmdb_env = lmdb.open(lmdb_name, readonly=True)
lmdb_txn = lmdb_env.begin()

cursor = lmdb_txn.cursor()

binlabels = {0:"background",1:"neutrino"}
classlabels = binlabels.keys()

input_shape = net.blobs["data"].data.shape
images_per_batch = 16
if input_shape[0]%images_per_batch!=0:
    print "Images per Batch must be multiple of shape. %d/%d=%d"%(input_shape[0],images_per_batch,input_shape[0]%images_per_batch)
    sys.exit(-1)


# ROOT data
print "[ENTER] to continue."
raw_input()


# setup output
out = rt.TFile("out_netanalysis.root", "RECREATE" )
herrmat = rt.TH2D("herrmat",";truth label;decision label",len(classlabels),0,len(classlabels),len(classlabels),0,len(classlabels))
hclassacc = rt.TH1D( "hclassacc", ";truth label;accuracy",len(classlabels),0,len(classlabels));
hclassfre = rt.TH1D( "hclassfreq", ";truth label;frequency",len(classlabels),0,len(classlabels));
hnuprob_nu      = rt.TH1D("hnuprob_nu",";prob",100,0,1)
hnuprob_cosmics = rt.TH1D("hnuprob_cosmics",";prob",100,0,1)
henergy = {}
henergy_miss = {}
for iclass in classlabels:
    henergy[binlabels[iclass]] = rt.TH1D( "henergy_%s_gev"%(binlabels[iclass]), "",50, 0,2.0 )
    henergy_miss[binlabels[iclass]] = rt.TH1D( "henergy_miss_%s_gev"%(binlabels[iclass]), "", 50, 0, 2.0 )

misslist = []
missdict = {}
totevents = 0.0
ibatch = 0
nbatches = 1000
correct  = 0.0
ncorrect_nu = 0
ncorrect_bg = 0
ntotal_nu = 0
ntotal_bg = 0

outofentries = False

# mean proto
fmean = open(mean_file,'rb')
mean_bin = fmean.read()
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(mean_bin)
mean_arr = np.array( caffe.io.blobproto_to_array(mean_blob) )
fmean.close()
#print mean_arr

data = np.zeros( input_shape, dtype=np.float32 )
input_labels = np.zeros( (input_shape[0],), dtype=np.float32 )
datum = caffe.proto.caffe_pb2.Datum()

resultlog = open('results.txt','w')

while not outofentries:
    print "batch ",ibatch," of ",nbatches
    keys = []
    nfilled = 0

    # we do multiple crops for each image
    ngroups_this_batch = 0
    for group in range( input_shape[0]/images_per_batch ):
        cursor.next()
        (key,raw_datum) = cursor.item()
        if key=='':
            outofentries = True
            break
        ngroups_this_batch += 1
        datum.ParseFromString(raw_datum)
        vec = datum_to_array(datum)
        keys.append(key)

        for n in range(0,images_per_batch):
            if nfilled>=input_shape[0]:
                break
            xoffset = int(np.random.rand()*(vec.shape[1]-input_shape[2]-1))
            yoffset = int(np.random.rand()*(vec.shape[2]-input_shape[3]-1))
            x1 = xoffset
            x2 = x1 + input_shape[2]
            y1 = yoffset
            y2 = y1 + input_shape[3]
            data[nfilled,:,:,:] = vec[:,x1:x2,y1:y2]-mean_arr[0,:,x1:x2,y1:y2]
            input_labels[nfilled] = datum.label
            nfilled += 1

    #print data[0,:,:,:]
    #raw_input()
    net.set_input_arrays( data, input_labels )

    net.forward()
    for group in range( ngroups_this_batch ):
        labels =  net.blobs["label"].data[group*images_per_batch:(group+1)*images_per_batch]
        scores = net.blobs["fc2"].data[group*images_per_batch:(group+1)*images_per_batch]
        probs = net.blobs["probt"].data[group*images_per_batch:(group+1)*images_per_batch]
        
        print labels[:,0,0,0]
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
        # Use Max
        #most_nu =np.argmax(probs[:,1])
        #decision = np.argmax(scores[most_nu])
        #ilabel = int(labels[most_nu])
        #prob = probs[most_nu]
        #score = scores[most_nu]
        hclassfre.Fill( ilabel )
        print "group ",group,":",labels,scores,probs
        print >> resultlog,key,decision,prob[1]

        totevents += 1.0
        print "label=",ilabel," vs. decision=",decision
        if ilabel==decision:
            correct += 1.0
            hclassacc.Fill( ilabel )
            if ilabel==0:
                ncorrect_bg+=1
            else:
                ncorrect_nu+=1
        else:
            print "Miss: ",key,ilabel,np.argmax(score)
            misslist.append( (binlabels[ilabel],key) )
            missdict[ (binlabels[ilabel],key) ] = {"key":key,"truth_label":ilabel,"decision":int(np.argmax(score)),"nuprob":prob[1]}
            herrmat.Fill( ilabel, decision )
        if ilabel==0:
            hnuprob_cosmics.Fill( prob[1] )
            ntotal_bg += 1
        else:
            hnuprob_nu.Fill( prob[1] )
            ntotal_nu += 1
    if totevents>0:
        print "running accuracy: ",correct/totevents
    ibatch += 1
    if ibatch>=nbatches:
        break
    #raw_input()

if ntotal_bg>0:
    print "Cosmic accuracy: ",float(ncorrect_bg)/float(ntotal_bg)
if ntotal_nu>0:
    print "Neutrino accuracy: ",float(ncorrect_nu)/float(ntotal_nu)

for miss in misslist:
    print miss, missdict[miss]

# properly normalize mistake matrix
for iclass in classlabels:
    tot = 0.0
    for jclass in classlabels:
        tot += herrmat.GetBinContent( iclass+1, jclass+1 )
    for jclass in classlabels:
        binval = herrmat.GetBinContent( iclass+1, jclass+1 )
        if tot>0:
            herrmat.SetBinContent( iclass+1, jclass+1, float(binval)/float(tot) )
        else:
            herrmat.SetBinContent( iclass+1, jclass+1, 0 )
    
out.Write()
resultlog.close()
