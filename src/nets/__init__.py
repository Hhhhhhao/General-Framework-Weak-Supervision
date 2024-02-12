from .wrn import wrn_28_2, wrn_28_8, wrn_34_10
from .wrn_var import wrn_var_37_2
from .resnet import resnet34, resnet50, resnet18, resnet50_pretrained, resnet18_pretrained
from .preact_resnet import preact_resnet18
from .inception_resnet import inception_resnet_v2
from .lenet import attn_lenet5, gated_attn_lenet5, lenet5, lenet5_c3

def get_model(model_name, num_classes, pretrained=False):
    model = eval(model_name)(num_classes=num_classes, pretrained=pretrained)
    return model