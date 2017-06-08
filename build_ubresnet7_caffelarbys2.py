import os,sys
import ROOT
from larcv import larcv
import layer_tools as lt
import caffe
from caffe import params as P
from caffe import layers as L

augment_data = True
use_batch_norm = True
use_dropout = True

def buildnet( inputdb, mean_file, batch_size, height, width, nchannels, net_type="train"):
    net = caffe.NetSpec()

    crop_size = -1
    if augment_data:
        crop_size = width

    train = False
    if net_type=="train":
        train = True

    data_layers,label = lt.data_layer_stacked( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=crop_size )

    # First conv  layer
    conv1 = lt.convolution_layer( net, data_layers[0], "conv1", "conv1", 16, 2, 7, 3, 0.05, addbatchnorm=True, train=train )
    pool1 = lt.pool_layer( net, conv1, "pool1", 5, 3 )

    resnet2  = lt.resnet_module( net, pool1,   "resnet2", 16, 3, 1, 1,8,16, use_batch_norm, train, "resnet2")
    resnet3  = lt.resnet_module( net, resnet2, "resnet3", 16, 3, 1, 1,8,16, use_batch_norm, train, "resnet3")
    resnet4  = lt.resnet_module( net, resnet3, "resnet4", 16, 3, 1, 1,8,32, use_batch_norm, train, "resnet4")
    
    resnet5  = lt.resnet_module( net, resnet4, "resnet5", 32, 3, 1, 1,8,32, use_batch_norm, train, "resnet5")
    resnet6  = lt.resnet_module( net, resnet5, "resnet6", 32, 3, 1, 1,8,32, use_batch_norm, train, "resnet6")
    resnet7  = lt.resnet_module( net, resnet6, "resnet7", 32, 3, 1, 1,16,64, use_batch_norm, train, "resnet7")

    resnet8  = lt.resnet_module( net, resnet7, "resnet8", 64,  3, 1, 1,16,64, use_batch_norm, train, "resnet8")
    resnet9  = lt.resnet_module( net, resnet8, "resnet9", 64, 3, 1, 1, 16,64, use_batch_norm, train, "resnet9")
    resnet10 = lt.resnet_module( net, resnet9, "resnet10", 64, 3, 1, 1,32,128, use_batch_norm, train, "resnet10")
        
    net.lastpool = lt.pool_layer( net, resnet10, "lastpool", 7, 1, P.Pooling.AVE )
    lastpool_layer = net.lastpool

    if use_dropout:
        net.lastpool_dropout = L.Dropout(net.lastpool,
                                         in_place=True,
                                         dropout_param=dict(dropout_ratio=0.5))
        lastpool_layer = net.lastpool_dropout
    

    fc2 = lt.final_fully_connect( net, lastpool_layer )
    
    if train:
        net.loss = L.SoftmaxWithLoss(fc2, net.label )
        net.acc = L.Accuracy(fc2,net.label)
    else:
        net.probt = L.Softmax( fc2 )
        net.acc = L.Accuracy(fc2,net.label)

    return net

if __name__ == "__main__":
    
    #traindb = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_train.db"
    #train_mean = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_train_mean.bin"
    #testdb = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test.db"
    #test_mean = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test_mean.bin"

    traindb = "/home/taritree/working/larbys/staged_data/ccqe_noisecut_mcc7_extbnb_train.db"
    train_mean = "/home/taritree/working/larbys/staged_data/ccqe_noisecut_mcc7_extbnb_train_mean.bin"
    testdb = "/home/taritree/working/larbys/staged_data/ccqe_noisecut_mcc7_ext_bnb_test.db"
    test_mean = "/home/taritree/working/larbys/staged_data/ccqe_noisecut_mcc7_ext_bnb_test_mean.bin"
    

    train_net = buildnet( traindb, train_mean, 1, 768, 768, 3, net_type="train"  )
    test_net  = buildnet( testdb,  test_mean, 1, 768, 768, 3, net_type="test"  )
    deploy_net  = buildnet( testdb,  test_mean, 1, 768, 768, 3, net_type="deploy"  )

    testout = open('ubresnet7_deep_test.prototxt','w')
    trainout = open('ubresnet7_deep_train.prototxt','w')
    deployout = open('ubresnet7_deep_deploy.prototxt','w')
    print >> testout, test_net.to_proto()
    print >> trainout, train_net.to_proto()
    print >> deployout, deploy_net.to_proto()
    testout.close()
    trainout.close()
    deployout.close()



