import os,sys
from caffe import layers as L
from caffe import params as P

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

def resnet_module( net, bot, name, ninput, kernel_size, stride, pad, bottleneck_nout, expand_nout, use_batch_norm, train, parname_stem ):
    if ninput!=expand_nout:
        bypass_conv = L.Convolution( bot,
                                     kernel_size=1,
                                     stride=1,
                                     num_output=expand_nout,
                                     pad=0,
                                     bias_term=False,
                                     weight_filler=dict(type="msra"),
                                     param=[dict(name="par_%s_bypass_conv_w"%(parname_stem))] )
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
    bottleneck_layer = L.Convolution(bot,num_output=bottleneck_nout,
                                     kernel_size=1,stride=1,pad=0,
                                     bias_term=False,weight_filler=dict(type="msra"),
                                     param=[dict(name="par_%s_bottleneck_conv_w"%(parname_stem))])
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
    conv_layer = L.Convolution(bottleneck_relu,num_output=bottleneck_nout,
                               kernel_size=3,stride=1,pad=1,
                               bias_term=False,
                               weight_filler=dict(type="msra"),
                               param=[dict(name="par_%s_conv_w"%(parname_stem))])
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
    expand_layer = L.Convolution(conv_relu,num_output=expand_nout,kernel_size=1,stride=1,pad=0,
                                 bias_term=False,
                                 weight_filler=dict(type="msra"),
                                 param=[dict(name="par_%s_expand_conv_w"%(parname_stem))])
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

def data_layer_trimese( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, slice_points, crop_size=-1 ):
    data, label = data_layer_stacked( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=crop_size )
    slices = L.Slice(data[0], ntop=3, name="data_trimese", slice_param=dict(axis=1, slice_point=slice_points))
    #for n,slice in enumerate(slices):
    #    net.__setattr__( slice, "data_plane%d"%(n) )

    return slices, label
    
def pool_layer( net, inputlayer, layername, kernel_size, stride, pooltype=P.Pooling.MAX ):
    pooll = L.Pooling(inputlayer, kernel_size=kernel_size, stride=stride, pool=pooltype)
    net.__setattr__( layername, pooll )
    return pooll

def slice_layer(net, layername, inputlayer, axis, slice_points):
    slices = L.Slice(inputlayer, ntop=3, name=layername, slice_param=dict(axis=axis, slice_point=slice_points))
    for n,slic in enumerate(slices):
	net.__setattr__(layername+"_%d"%(n), slic)
    return slices
