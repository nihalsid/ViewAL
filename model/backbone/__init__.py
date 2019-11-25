from model.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, output_stride, BatchNorm, mc_dropout):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, mc_dropout=mc_dropout)
    else:
        raise NotImplementedError
