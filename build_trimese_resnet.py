import os,sys
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

    data_layers,label = lt.data_layer_trimese( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, [4,8], crop_size=768 )

    # First conv  layer
    branch_ends = []
    for n,layer in enumerate(data_layers):
        conv1 = lt.convolution_layer( net, layer, "plane%d_conv1"%(n), "tri_conv1", 32, 2, 7, 3, 0.05, addbatchnorm=True, train=train )
        pool1 = lt.pool_layer( net, conv1, "plane%d_pool1"%(n), 3, 1 )

        conv2 = lt.convolution_layer( net, pool1, "plane%d_conv2"%(n), "tri_conv2", 16, 2, 3, 3, 0.05, addbatchnorm=True, train=train )
        
        conv3 = lt.convolution_layer( net, conv2, "plane%d_conv3"%(n), "tri_conv3", 16, 2, 3, 3, 0.05, addbatchnorm=True, train=train )

        pool3 = lt.pool_layer( net, conv3, "plane%d_pool3"%(n), 3, 1 )

        branch_ends.append( pool3 )
        
    concat = lt.concat_layer( net, "mergeplanes", *branch_ends )


    resnet1  = lt.resnet_module( net, concat,  "resnet1", 16*3, 3, 1, 1,8,16, use_batch_norm, train)
    resnet2  = lt.resnet_module( net, resnet1, "resnet2", 16, 3, 1, 1,8,16, use_batch_norm, train)
    resnet3  = lt.resnet_module( net, resnet2, "resnet3", 16, 3, 1, 1,8,32, use_batch_norm, train)
    
    resnet4  = lt.resnet_module( net, resnet3, "resnet4", 32, 3, 1, 1,8,32, use_batch_norm, train)
    resnet5  = lt.resnet_module( net, resnet4, "resnet5", 32, 3, 1, 1,8,32, use_batch_norm, train)
    resnet6  = lt.resnet_module( net, resnet5, "resnet6", 32, 3, 1, 1,16,64, use_batch_norm, train)

    resnet7  = lt.resnet_module( net, resnet6, "resnet7", 64,  3, 1, 1,16,64, use_batch_norm, train)
    resnet8  = lt.resnet_module( net, resnet7, "resnet8", 64, 3, 1, 1, 16,64, use_batch_norm, train)
    resnet9  = lt.resnet_module( net, resnet8, "resnet9", 64, 3, 1, 1,32,128, use_batch_norm, train)
        
    net.lastpool = lt.pool_layer( net, resnet9, "lastpool", 7, 1, P.Pooling.AVE )
    lastpool_layer = net.lastpool
    
    if use_dropout:
        net.lastpool_dropout = L.Dropout(net.lastpool,
                                         in_place=True,
                                         dropout_param=dict(dropout_ratio=0.5))
        lastpool_layer = net.lastpool_dropout
    
    fc1 = lt.final_fully_connect( net, lastpool_layer, nclasses=256 )
    fc2 = lt.final_fully_connect( net, fc1, nclasses=4096 )
    fc3 = lt.final_fully_connect( net, fc2, nclasses=2 )
    
    if train:
        net.loss = L.SoftmaxWithLoss(fc3, net.label )
        net.acc = L.Accuracy(fc3,net.label)
    else:
        net.probt = L.Softmax( fc3 )
        net.acc = L.Accuracy(fc3,net.label)

    return net

if __name__ == "__main__":
    
    traindb    = "/mnt/raid0/taritree/test_data/ccqe_supported_images_train.db"
    train_mean = "/mnt/raid0/taritree/test_data/ccqe_supported_images_train_mean.bin"
    testdb     = "/mnt/raid0/taritree/test_data/ccqe_supported_images_test.db"
    test_mean  = "/mnt/raid0/taritree/test_data/ccqe_supported_images_test_mean.bin"
    

    train_net = buildnet( traindb, train_mean, 8, 768, 768, 3, net_type="train"  )
    test_net  = buildnet( testdb,   test_mean, 1, 768, 768, 3, net_type="test"  )
    deploy_net  = buildnet( testdb, test_mean, 1, 768, 768, 3, net_type="deploy"  )

    testout   = open('ub_trimese_resnet_test.prototxt','w')
    trainout  = open('ub_trimese_resnet_train.prototxt','w')
    deployout = open('ub_trimese_resnet_deploy.prototxt','w')
    print >> testout, test_net.to_proto()
    print >> trainout, train_net.to_proto()
    print >> deployout, deploy_net.to_proto()
    testout.close()
    trainout.close()
    deployout.close()



