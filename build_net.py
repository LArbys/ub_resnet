import os,sys
import caffe
from caffe import layers as L
from caffe import params as P

use_batch_norm = True
use_dropout = True
augment_data = True


def convolution_layer( net, input_layer, layername_stem, parname_stem, noutputs, stride, kernel_size, pad, init_bias, addbatchnorm=True, train=True ):
    conv = L.Convolution( input_layer, 
                          kernel_size=kernel_size,
                          stride=stride,
                          pad=pad,
                          num_output=noutputs,
                          weight_filler=dict(type="msra"),
                          bias_filler=dict(type="constant",value=init_bias),
                          param=[dict(name="par_%s_conv_w"%(parname_stem)),dict(name="par_%s_conv_b"%(parname_stem))] )
    net.__setattr__( layername_stem+"_conv", conv )
    if addbatchnorm:
        if train:
            conv_bn = L.BatchNorm( conv, in_place=True, batch_norm_param=dict(use_global_stats=False),param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            conv_bn = L.BatchNorm( conv,in_place=True,batch_norm_param=dict(use_global_stats=True))
        conv_scale = L.Scale( conv_bn, in_place=True, scale_param=dict(bias_term=True))
        conv_relu  = L.ReLU(conv_scale,in_place=True)
        net.__setattr__( layername_stem+"_bn", conv_bn )
        net.__setattr__( layername_stem+"_scale", conv_scale )
        net.__setattr__( layername_stem+"_relu", conv_relu )
        nxtlayer = conv_relu
    else:
        conv_relu  = L.ReLU( conv, in_place=True )
        net.__setattr__( layername_stem+"_relu", conv_relu )
        nxtlayer = conv_relu
    return nxtlayer

def concat_layer( net, layername, *bots ):
    convat = L.Concat(*bots, concat_param=dict(axis=1))
    net.__setattr__( "%s_concat"%(layername), convat )
    return convat
        
    
def final_fully_connect( net, bot, nclasses=2 ):
    net.fc2 = L.InnerProduct( bot, num_output=nclasses, weight_filler=dict(type='msra'))
    return net.fc2

def resnet_module( net, bot, name, ninput, kernel_size, stride, pad, bottleneck_nout, expand_nout, train ):
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

def data_layer_stacked( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=-1 ):
    transform_pars = {"mean_file":mean_file,
                      "mirror":False}
    if crop_size>0:
        transform_pars["crop_size"] = crop_size
    if net_type in ["train","test"]:
        net.data, net.label = L.Data(ntop=2,backend=P.Data.LMDB,source=inputdb,batch_size=batch_size,transform_param=transform_pars)
    elif net_type=="deploy":
        net.data, net.label = L.MemoryData(ntop=2,batch_size=batch_size, height = height, width = width, channels = nchannels)
    return [net.data], net.label

def data_layer_trimese( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=-1 ):
    data, label = data_layer_stacked( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=crop_size )
    slices = L.Slice(data[0], ntop=3, name="data_trimese", slice_param=dict(axis=1, slice_point=[1,2]))
    return slices, label
    
def pool_layer( net, inputlayer, layername, kernel_size, stride ):
    pooll = L.Pooling(inputlayer, kernel_size=kernel_size, stride=stride, pool=P.Pooling.MAX)
    net.__setattr__( layername, pooll )
    return pooll

def buildnet( inputdb, mean_file, batch_size, height, width, nchannels, net_type="train", trimese=True  ):
    net = caffe.NetSpec()

    crop_size = -1
    if augment_data:
        crop_size = width

    train = False
    if net_type=="train":
        train = True

    if trimese:
        data_layers, label = data_layer_trimese( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=crop_size )
    else:
        data_layers,label = data_layer_stacked( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=crop_size )

    tri = []
    for n,l in enumerate(data_layers):
        # no batch norm in first layers, reduces the required number of parameters

        # First conv  layer
        conv1 = convolution_layer( net, l, "plane%d_conv1"%(n), "tri_conv1", 16, 2, 7, 3, 0.05, addbatchnorm=True, train=train )
        pool1 = pool_layer( net, conv1, "plane%d_pool1"%(n), 3, 2 )
        # resnet layers
        resnet2 = resnet_module( net, pool1,   "plane%d_resnet2"%(n), 16, 3, 1, 1,8,32, train)
        resnet3 = resnet_module( net, resnet2, "plane%d_resnet3"%(n), 32, 3, 1, 1,8,32, train)
        resnet4 = resnet_module( net, resnet3, "plane%d_resnet4"%(n), 32, 3, 1, 1,8,32, train)
        
        tri.append( resnet4 )
        
    
    concat = concat_layer( net, "mergeplanes", *tri )

    #conv1_top = addFirstConv(net,32,train)
    #conv1_top = ( net, 32, train )
    #sys.exit(-1)
    #net.pool1 = L.Pooling(conv1_top, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    #resnet2 = resnet_module( "resnet2", net, net.pool1, 32, 3, 1, 1,16,32, train)
    #resnet3 = resnet_module( "resnet3", net, resnet2, 32, 3, 1, 1,16,32,train)
    #resnet4 = resnet_module( "resnet4", net, resnet3, 32, 3, 1, 1,16,32,train)

    #resnet5 = resnet_module( "resnet5", net, resnet4, 32, 3, 1, 1,16,64,train)
    #resnet6 = resnet_module( "resnet6", net, resnet5, 64, 3, 1, 1,16,64,train)
    #resnet7 = resnet_module( "resnet7", net, resnet6, 64, 3, 1, 1,16,64,train)

    #resnet8 = resnet_module( "resnet8", net, resnet7, 64, 3, 1, 1,32,128,train)
    #resnet9 = resnet_module( "resnet9", net, resnet8, 128, 3, 1, 1,32,128,train)
    #resnet10 = resnet_module( "resnet10", net, resnet9, 128, 3, 1, 1,32,128,train)

    net.lastpool = L.Pooling( concat, kernel_size=7, stride=1, pool=P.Pooling.AVE )
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
    
    traindb = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_train.db"
    train_mean = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_train_mean.bin"
    testdb = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test.db"
    test_mean = "/mnt/disk0/taritree/larbys/prepared_lmdb/ccqe_combined_extbnbcosmic_mcc7nu_test_mean.bin"
    

    tri = True
    train_net = buildnet( traindb, train_mean, 1, 768, 768, 3, net_type="train", trimese=tri )
    test_net  = buildnet( testdb,  test_mean, 1, 768, 768, 3, net_type="test", trimese=tri )
    deploy_net  = buildnet( testdb,  test_mean, 1, 768, 768, 3, net_type="deploy", trimese=tri )

    testout = open('test_v3.prototxt','w')
    trainout = open('train_v3.prototxt','w')
    deployout = open('deploy_v3.prototxt','w')
    print >> testout, test_net.to_proto()
    print >> trainout, train_net.to_proto()
    print >> deployout, deploy_net.to_proto()
    testout.close()
    trainout.close()
    deployout.close()
