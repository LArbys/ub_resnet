import os,sys
import caffe
from caffe import layers as L
from caffe import params as P

use_batch_norm = True
use_dropout = True
augment_data = True

def addFirstConv( net, noutput, train ):
    net.conv1 = L.Convolution(net.data,
                              kernel_size=7,
                              stride=2,
                              pad=3,
                              num_output=noutput,
                              weight_filler=dict(type="msra"),
                              #group=1,
                              bias_filler=dict(type="constant",value=0.05))
    #param=[dict(name="conv1_w"),dict(name="conv1_b")] )
    if use_batch_norm:
        if train:
            net.conv1_bn = L.BatchNorm(net.conv1,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                       param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            net.conv1_bn = L.BatchNorm(net.conv1,in_place=True,batch_norm_param=dict(use_global_stats=True))
        net.conv1_scale = L.Scale(net.conv1_bn,
                                  in_place=True,
                                  scale_param=dict(bias_term=True))
        net.conv1_relu = L.ReLU(net.conv1_scale,in_place=True)
    else:
        net.conv1_relu = L.ReLU(net.conv1,in_place=True)
    #net.conv1_dropout = L.Dropout(net.conv1_scale,
    #                              in_place=True,
    #                              dropout_param=dict(dropout_ratio=0.5))
    return net.conv1_relu
    
def final_fully_connect( net, bot, nclasses=2 ):
    net.fc2 = L.InnerProduct( bot, num_output=nclasses, weight_filler=dict(type='msra'))
    return net.fc2

def resnet_module( name, net, bot, ninput, kernel_size, stride, pad, bottleneck_nout, expand_nout, train ):
    if ninput!=expand_nout:
        bypass_conv = L.Convolution( bot,
                                     kernel_size=1,
                                     stride=1,
                                     num_output=expand_nout,
                                     pad=0,
                                     bias_term=False,
                                     weight_filler=dict(type="msra") )
        if use_batch_norm:
            if train:
                bypass_bn = L.BatchNorm(bypass_conv,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                        param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
            else:
                bypass_bn = L.BatchNorm(bypass_conv,in_place=True,batch_norm_param=dict(use_global_stats=True))
            bypass_scale = L.Scale(bypass_bn,in_place=True,scale_param=dict(bias_term=True))
            net.__setattr__(name+"_bypass",bypass_conv)
            net.__setattr__(name+"_bypass_bn",bypass_bn)
            net.__setattr__(name+"_bypass_scale",bypass_scale)
        else:
            net.__setattr__(name+"_bypass",bypass_conv)
        bypass_layer = bypass_conv
    else:
        bypass_layer  = bot

    # bottle neck
    bottleneck_layer = L.Convolution(bot,num_output=bottleneck_nout,kernel_size=1,stride=1,pad=0,bias_term=False,weight_filler=dict(type="msra"))
    if use_batch_norm:
        if train:
            bottleneck_bn    = L.BatchNorm(bottleneck_layer,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                           param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            bottleneck_bn    = L.BatchNorm(bottleneck_layer,in_place=True,batch_norm_param=dict(use_global_stats=True))
        bottleneck_scale = L.Scale(bottleneck_bn,in_place=True,scale_param=dict(bias_term=True))
        bottleneck_relu  = L.ReLU(bottleneck_scale,in_place=True)
    else:
        bottleneck_relu  = L.ReLU(bottleneck_layer,in_place=True)
    net.__setattr__(name+"_btlnk",bottleneck_layer)
    if use_batch_norm:
        net.__setattr__(name+"_btlnk_bn",bottleneck_bn)
        net.__setattr__(name+"_btlnk_scale",bottleneck_scale)
    net.__setattr__(name+"_btlnk_relu",bottleneck_relu)

    # conv
    conv_layer = L.Convolution(bottleneck_relu,num_output=bottleneck_nout,kernel_size=3,stride=1,pad=1,bias_term=False,weight_filler=dict(type="msra"))
    if use_batch_norm:
        if train:
            conv_bn    = L.BatchNorm(conv_layer,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                     param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            conv_bn    = L.BatchNorm(conv_layer,in_place=True,batch_norm_param=dict(use_global_stats=True))
        conv_scale = L.Scale(conv_bn,in_place=True,scale_param=dict(bias_term=True))
        conv_relu  = L.ReLU(conv_scale,in_place=True)
    else:
        conv_relu  = L.ReLU(conv_layer,in_place=True)
    net.__setattr__(name+"_conv",conv_layer)
    if use_batch_norm:
        net.__setattr__(name+"_conv_bn",conv_bn)
        net.__setattr__(name+"_conv_scale",conv_scale)
    net.__setattr__(name+"_conv_relu",conv_relu)

    # expand
    expand_layer = L.Convolution(conv_relu,num_output=expand_nout,kernel_size=1,stride=1,pad=0,bias_term=False,weight_filler=dict(type="msra"))
    ex_last_layer = expand_layer
    if use_batch_norm:
        if train:
            expand_bn    = L.BatchNorm(expand_layer,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                       param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            expand_bn    = L.BatchNorm(expand_layer,in_place=True,batch_norm_param=dict(use_global_stats=True))
        expand_scale = L.Scale(expand_bn,in_place=True,scale_param=dict(bias_term=True))
        ex_last_layer = expand_scale
    net.__setattr__(name+"_expnd",expand_layer)
    if use_batch_norm:
        net.__setattr__(name+"_expnd_bn",expand_bn)
        net.__setattr__(name+"_expnd_scale",expand_scale)

    # Eltwise
    elt_layer = L.Eltwise(bypass_layer,ex_last_layer, eltwise_param=dict(operation=P.Eltwise.SUM))
    elt_relu  = L.ReLU( elt_layer,in_place=True)
    net.__setattr__(name+"_eltwise",elt_layer)
    net.__setattr__(name+"_eltwise_relu",elt_relu)
    

    return elt_relu
                                      
def batchnorm_layer():
    pass

def buildnet( inputdb, mean_file, batch_size, train=True ):
    net = caffe.NetSpec()
    transform_pars = {"mean_file":mean_file,
                      "mirror":False}
    if augment_data:
        transform_pars["crop_size"] = 320
                      
    net.data, net.label = L.Data(ntop=2,backend=P.Data.LMDB,
                                 source=inputdb, batch_size=batch_size,
                                 transform_param=transform_pars )
    conv1_top = addFirstConv(net,32,train)
    net.pool1 = L.Pooling(conv1_top, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    resnet2 = resnet_module( "resnet2", net, net.pool1, 32, 3, 1, 1,16,32, train)
    resnet3 = resnet_module( "resnet3", net, resnet2, 32, 3, 1, 1,16,32,train)
    resnet4 = resnet_module( "resnet4", net, resnet3, 32, 3, 1, 1,16,32,train)

    resnet5 = resnet_module( "resnet5", net, resnet4, 32, 3, 1, 1,16,64,train)
    resnet6 = resnet_module( "resnet6", net, resnet5, 64, 3, 1, 1,16,64,train)
    resnet7 = resnet_module( "resnet7", net, resnet6, 64, 3, 1, 1,16,64,train)

    resnet8 = resnet_module( "resnet8", net, resnet7, 64, 3, 1, 1,32,128,train)
    resnet9 = resnet_module( "resnet9", net, resnet8, 128, 3, 1, 1,32,128,train)
    resnet10 = resnet_module( "resnet10", net, resnet9, 128, 3, 1, 1,32,128,train)

    net.lastpool = L.Pooling( resnet10, kernel_size=7, stride=1, pool=P.Pooling.AVE )
    lastpool_layer = net.lastpool
    if use_dropout:
        net.lastpool_dropout = L.Dropout(net.lastpool,
                                         in_place=True,
                                         dropout_param=dict(dropout_ratio=0.5))
        lastpool_layer = net.lastpool_dropout


    fc2 = final_fully_connect( net, lastpool_layer )
    
    if train:
        net.loss = L.SoftmaxWithLoss(fc2, net.label )
        net.acc = L.Accuracy(fc2,net.label)
    else:
        net.probt = L.Softmax( fc2 )
        net.acc = L.Accuracy(fc2,net.label)
    return net

if __name__ == "__main__":
    traindb = "/home/taritree/working/larbys/staged_data/resized_traindata_combinedbnbcosmics.db"
    train_mean = "/home/taritree/working/larbys/staged_data/resized_traindata_combinedbnbcosmics_mean.bin"
    #traindb = "/mnt/disk0/taritree/larbys/prepared_lmdb/overfit_resized_combinedbnbcosmics.db"
    #train_mean = "/mnt/disk0/taritree/larbys/prepared_lmdb/overfit_resized_combinedbnbcosmics_mean.bin"

    testdb = "/home/taritree/working/larbys/staged_data/resized_testdata_combinedbnbcosmics.db"
    test_mean = "/home/taritree/working/larbys/staged_data/resized_testdata_combinedbnbcosmics_mean.bin"
    #testdb = "/mnt/disk0/taritree/larbys/prepared_lmdb/overfit_resized_combinedbnbcosmics_test.db"
    #test_mean = "/mnt/disk0/taritree/larbys/prepared_lmdb/overfit_resized_combinedbnbcosmics_mean_test.bin"

    train_net = buildnet( traindb, train_mean, 32, train=True )
    test_net  = buildnet( testdb,  test_mean, 16, train=False )

    testout = open('test.prototxt','w')
    trainout = open('train.prototxt','w')
    print >> testout, test_net.to_proto()
    print >> trainout, train_net.to_proto()
    testout.close()
    trainout.close()
