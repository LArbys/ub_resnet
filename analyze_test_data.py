import caffe
import numpy as np
import ROOT as rt
import lmdb
from math import log

caffe.set_mode_gpu()

test_prototxt = "training_attempts/0017/test.prototxt"
data_prototxt = "training_attempts/0017/data.prototxt"
model = "training_attempts/0017/snapshot_rmsprop_iter_50000.caffemodel"
train_data = "/home/taritree/working/larbys/staged_data/resized_traindata_combinedbnbcosmics.db"
#test_data  = "/home/taritree/working/larbys/staged_data/resized_testdata_combinedbnbcosmics.db"
test_data = "/home/taritree/working/larbys/staged_data/resized_databnb.db"
root_dirs = {"eminus":"/mnt/disk0/taritree/larbys/new/single_eminus/rootfiles",
             "muminus":"/mnt/disk0/taritree/larbys/new/single_proton/rootfiles",
             "proton":"/mnt/disk0/taritree/larbys/new/single_muminus/rootfiles",
             "pizero":"/mnt/disk0/taritree/larbys/new/single_pizero_bnblike/rootfiles"}

prototxt = data_prototxt
net = caffe.Net( prototxt, model, caffe.TEST )

lmdb_name = test_data
lmdb_env = lmdb.open(lmdb_name, readonly=True)
lmdb_txn = lmdb_env.begin()

cursor = lmdb_txn.cursor()

batchsize = 16
binlabels = {0:"background",1:"neutrino"}
classlabels = binlabels.keys()

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
hll_nu = rt.TH1D("hll_nu",";prob",100,-10,10)
hll_bg = rt.TH1D("hll_bg",";prob",100,-10,10)
henergy = {}
henergy_miss = {}
for iclass in classlabels:
    henergy[binlabels[iclass]] = rt.TH1D( "henergy_%s_gev"%(binlabels[iclass]), "",50, 0,2.0 )
    henergy_miss[binlabels[iclass]] = rt.TH1D( "henergy_miss_%s_gev"%(binlabels[iclass]), "", 50, 0, 2.0 )

misslist = []
missdict = {}
totevents = 0.0
ibatch = 0
nbatches = 800
ncorrect_nu = 0
ncorrect_bg = 0
ntotal_nu = 0
ntotal_bg = 0
while ibatch<nbatches:
    print "batch ",ibatch," of ",nbatches
    keys = []
    for iimg in range(0,batchsize):
        cursor.next()
        (key,raw_datum) = cursor.item()
        #datum = caffe.proto.caffe_pb2.Datum()
        #datum.ParseFromString(raw_datum)
        #feature = caffe.io.datum_to_array(datum)
        #label = datum.label
        #labels.append( label )
        #batch_images.append( feature[:,10:210,10:210] )
        keys.append(key)

    net.forward()
    labels =  net.blobs["label"].data
    scores = net.blobs["fc2"].data
    probs = net.blobs["probt"].data
    correct = 0.0
    nevals = 0.0
    #totevents += float( len(scores) )
    #print scores
    #print prob
    #print labels
    for label,score,prob,key in zip(labels,scores,probs,keys):
        #print label,score
        ilabel = int(label)
        decision = np.argmax(score)
        #print decision," ",ilabel
        hclassfre.Fill( ilabel )
        nevals += 1.0
        #print prob[0],prob[1]
        if prob[0]==1:
            ll = -99
        elif prob[1]==1:
            ll = 99
        else:
            ll = log(prob[1]) - log(prob[0])
            if ll <-100:
                ll = -99
            elif ll>100:
                ll = 99
        if ilabel==decision:
            correct += 1.0
            hclassacc.Fill( ilabel )
            if ilabel==0:
                ncorrect_bg+=1
            else:
                ncorrect_nu+=1
        else:
            #print "Miss: ",key,label,np.argmax(score)
            misslist.append( (binlabels[ilabel],key) )
            missdict[ (binlabels[ilabel],key) ] = {"key":key,"truth_label":int(label),"decision":int(np.argmax(score)),"nuprob":prob[1]}
            herrmat.Fill( ilabel, decision )
            #if evtbytes>0:
            #    henergy_miss[binlabels[ilabel]].Fill( bbtree.Enu )
        herrmat.Fill( ilabel, decision )
        if ilabel==0:
            hnuprob_cosmics.Fill( prob[1] )
            hll_bg.Fill( ll )
            ntotal_bg += 1
        else:
            hnuprob_nu.Fill( prob[1] )
            hll_nu.Fill( ll )
            ntotal_nu += 1
    print "accuracy: ",correct/nevals
    ibatch += 1
    #raw_input()

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
    
