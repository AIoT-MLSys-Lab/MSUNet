"""
This module defines an API for counting parameters and operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from timm.models import create_model
from validate import alpha
from timm.models.gen_efficientnet import InvertedResidual, SqueezeExcite, DepthwiseSeparableConv, Swish_Module, Sigmoid_Module, ReLU_Module
from timm.models.adaptive_avgmax_pool import SelectAdaptivePool2d
import argparse
from apex import amp

assert alpha == 0.67749, 'Alpha is not 0.67749'

parser = argparse.ArgumentParser()
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--binarizable', type=str, default='T', help='Using binary (B) or ternary (T)')
parser.add_argument('--amp', action='store_true', default=False, help='use NVIDIA amp for mixed precision training')
parser.add_argument('--opt-level', type=str, default='O1', help='Apex opt-level, "01"(conservative), "03"(Pure FP16)')


args = parser.parse_args()
binarizable = args.binarizable

def get_denominator(m, input, output):
    check_type_list = [input[0], output]
    if hasattr(m, 'weight'):
        if hasattr(m, '_get_weight'):
            assert m._get_weight('weight')[1].type() == m.weight.type(), "Multiplier type doesn't match the stored weight type"
            weight = m._get_weight('weight')[1] # multiplier
        else:
            weight = m.weight
        check_type_list.append(weight)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                assert weight.type() == m.bias.type(), "Weight type doesn't math bias type"
    if hasattr(m, 'running_mean'):
        check_type_list.append(m.running_mean)
    if hasattr(m, 'running_var'):
        check_type_list.append(m.running_var)

    if all([isinstance(v,torch.cuda.HalfTensor) for v in check_type_list]):
        # denominator if using Half precision
        return 2
    elif any([isinstance(v,torch.cuda.FloatTensor) for v in check_type_list]):
        # No denominator, if any of the input, output or weight is FloatTensor
        return 1
    else:
        raise RuntimeError('Unrecognized types')

def count_hook(model):
    '''
    Register a forward hook to each module of the model.
    The hook will be called every time when forward is called.
    The hook is responsible for calculating the number of parameters and flops for each base module.
    :param model: The MSUNet model
    '''
    def count_ops(m, input, output):
        '''
        :param m:       Low level and high level modules.
                        The low level modules are nn.Conv2d, nn.Linear, nn.BatchNorm2d, etc.
                        The high level modules are InvertedResidual, SqueezeExcite, etc.
        :param input:   The input features of the module
        :param output:  The output features of the module

        ## Global parameters
        :param param_count:         Accumulation of number of parameters in all modules
        :param flop_mults:          Accumulation of number of multiplications in all modules
        :param flop_adds:           Accumulation of number of additions in all modules
        :param module_statistics:   A dictionary that stores the module, #params and #flops
        :param module_skipped:      A list of modules that do not contribute to #params and #flops,
                                    which are mainly high level modules like nn.Sequential

        ## Local parameters
        :param local_param_count:   Number of parameters in module m
        :param local_flop_adds:     Number of additions in module m
        :param local_flop_mults:    Number of multiplications in module m
        :param c_in:                Number of input channels
        :param c_out:               Number of output channels
        :param k:                   Kernel size
        :param sparsity:            The sparsity of weights
        :param bd:                  Quantization denominator, which is 32 for binary weights and 1 for full precision weights
                                    This denominator is only used in ternary/binary nn.Conv2d layers
        :param d:                   Quantization denominator, which is 2 for FP16 (HalfTensor) and 1 for FP32 (FloatTensor) operation
        '''
        global param_count, flop_mults, flop_adds
        global module_statistics, module_skipped
        local_param_count, local_flop_adds, local_flop_mults = 0, 0, 0
        # For nn.Conv2d:    d is the quantization denominator for non-ternary/binary weights and operations
        #                   bd is the quantization denominator for ternary/binary weights and operations
        # For other modules: d is the quantization denominator for all the weights and operations
        # Note that the additions inside GEMM are considered FP32 additions
        d = get_denominator(m, input, output)


        if isinstance(m, nn.Conv2d):
            # Normal convolution, depthwise convolution, and 1x1 pointwise convolution,
            # with sparse and/or ternary/binary weights are all handled in this block.

            c_out, c_in, k, k2 = m.weight.size()
            # Square kerenl expected
            assert k == k2, 'The kernel is not square.'


            if hasattr(m, '_get_weight'):
                # The module having _get_weight attributed is Ternarized.

                # Ternary weight is considered as sparse binary weights,
                # so we use a quantization denominator 32 for multiplication and storage.
                bd = 32 # denominator for binary mults and parameters
                if binarizable == 'T':
                    # Using ternary quantization
                    #print('Using Ternary weights')

                    # Since ternary weights are considered as sparse binary weights,
                    # we do have to store a bit mask to represent sparsity.
                    local_param_count += c_out * c_in * k * k / 32
                    sparsity = (m._get_weight('weight')[0].numel() - m._get_weight('weight')[0].nonzero().size(0)) / m._get_weight('weight')[0].numel()

                    # Since our ternary/binary weights are scaled by a global factor in each layer,
                    # we do have to store a FP32/FP16 digit to represent it.
                    local_param_count += 1 / d # The scale factor
                elif binarizable == 'B':
                    # Using binary quantization
                    # Although we support binary quantization, our we prefer to use ternary quantization.
                    #print('Using Binary weights')
                    # The FP32/FP16 scale factor
                    local_param_count += 1 / d
                    sparsity = 0
                else:
                    raise ValueError('Option args.binarizable is incorrect')

                # Since our ternary/binary weights are scaled by a global factor, sqrt(M), in each layer,
                # which can be considered as multiplying a scale factor on the output of the sparse binary convolution.
                # We count it as FP32/FP16 multiplication on the output.
                local_flop_mults += np.prod(output.size()) / d

            else:
                # No binarization/ternarization is performed
                # However, we can still choose from full precision (FP32) or half precision (FP16)
                # Therefore, denominator for mults, adds and parameter storage all equal to d
                bd = d

                # Some layers are sparsed, we count those layers that have sparsity > 0.5
                sparsity_ = (m.weight.numel() - m.weight.nonzero().size(0)) / m.weight.numel()
                # The layers with sparsity < 0.4 does not count
                if sparsity_ < 0.4:
                    sparsity = 0
                # 6 squeeze-excitation conv layers and most dw-conv layers have sparsity > 0.4
                else:
                    sparsity = sparsity_
                    # The bit mask for sparse weights
                    local_param_count += c_out * c_in * k * k / 32

            if m.groups is not None:
                if m.groups != 1:
                    assert m.groups==c_out and c_in==1, 'm.groups is incorrect'



            # Number of parameters
            # For sparse parameters:                sparsity > 0
            # For dense parameters:                 sparsity=0
            # For 1-bit binary parameters:          bd==32
            # For 32-bit full precision parameters: bd==1
            # For 16-bit half precision parameters: bd==d==2
            # For depthwise convolution:            c_in==1
            local_param_count += c_out * c_in * k * k / bd * (1-sparsity)

            # Number of multiplications in convolution
            # For sparse multiplication:                sparsity > 0
            # For dense multiplication:                 sparsity=0
            # For 1-bit binary multiplication:          bd==32
            # For 32-bit full precision multiplication: bd==1
            # For 16-bit half precision parameters:     bd==d==2
            # For depthwise convolution:                c_in==1
            local_flop_mults += (k * k * c_in) * (1-sparsity) * np.prod(output.size()) / bd

            # Number of full precision (32-bit) addition in convolution
            local_flop_adds += (k * k * c_in * (1-sparsity) - 1) * np.prod(output.size())

            # The parameters and additions for the bias
            # For 32-bit full precision parameters:    d==1
            # For 16-bit half precision parameters:    d==2
            # Note that additions are in 32-bit full precision
            if m.bias is not None:
                local_param_count += c_out / d
                local_flop_adds += np.prod(output.size())

            # Adding the local counting to the global counting
            param_count += local_param_count
            flop_adds += local_flop_adds
            flop_mults += local_flop_mults
            module_statistics[id(m)] = (m, [local_param_count, local_flop_adds, local_flop_mults])


        elif isinstance(m, nn.Linear):
            c_out, c_in = m.weight.size()
            local_param_count += c_in * c_out / d
            local_flop_mults += c_in * np.prod(output.size()) / d
            local_flop_adds += (c_in - 1) * np.prod(output.size())

            param_count += local_param_count
            flop_mults += local_flop_mults
            flop_adds += local_flop_adds
            module_statistics[id(m)] = (m, [local_param_count, local_flop_adds, local_flop_mults])

        elif isinstance(m, nn.BatchNorm2d):
            using_FP16_for_conv = args.amp and args.opt_level != 'O0'
            if d == 1 and using_FP16_for_conv:
                # BatchNorm2d is performed with FP32 precision as indicated by d==1
                # Conv2d is performed with FP16 as indicated using_FP16_for_conv==True
                # We cannot merge BatchNorm2d with Conv2d, since they are performed with different precision
                print('Batchnorm cannot be merged.')
                local_param_count += 2 * input[0].size(0) * input[0].size(1) / d
                local_flop_adds += 1 * np.prod(input[0].size()) / d
                local_flop_mults += 1 * np.prod(input[0].size()) / d

            elif (d == 1 and not using_FP16_for_conv) or (d==2 and using_FP16_for_conv):
                print('Batchnorm can be merged')
                # BatchNorm2d and Conv2d are preformed using the same precision (FP32 or FP16)
                # We can then merge BatchNorm2d with Conv2d since they are performed with the same precision
                # However, because the Conv2d before BatchNorm2d does not supply bias term,
                # we can only merge the multiplication part, but not the addition part.
                # We then need to calculate the parameters and flops related to the additions in BatchNorm2d layers explicitly.
                local_param_count += 1*input[0].size(0) * input[0].size(1) / d # The parameters related to additions, which cannot
                                                                               # be merged to the non-existing bias of previous conv-layer
                local_flop_adds += 1*np.prod(input[0].size()) / d # The flops related to additions,
                                                                  # which cannot be merged to non-existing bias of previous conv-layer
            else:
                raise RuntimeError("Unsupported quantization scheme for batchnorm")

            param_count += local_param_count
            flop_mults += local_flop_mults
            flop_adds += local_flop_adds
            module_statistics[id(m)] = (m, [local_param_count, local_flop_adds, local_flop_mults])

        elif isinstance(m, nn.ReLU) or isinstance(m, ReLU_Module):
            flop_mults += np.prod(input[0].size()) / d

            module_statistics[id(m)] = (m, [0, 0, np.prod(input[0].size()) / d])

        elif isinstance(m, Swish_Module):
            flop_mults += 3 * np.prod(input[0].size()) / d
            flop_adds += np.prod(input[0].size()) / d

            module_statistics[id(m)] = (m, [0, np.prod(input[0].size()) / d, 3 * np.prod(input[0].size()) / d])

        elif isinstance(m, nn.Sigmoid) or isinstance(m, Sigmoid_Module):
            flop_mults += 2 * np.prod(input[0].size()) / d
            flop_adds += np.prod(input[0].size()) / d

            module_statistics[id(m)] = (m, [0, np.prod(input[0].size()) / d, 2 * np.prod(input[0].size()) / d])


        elif isinstance(m, nn.AvgPool2d):
            if isinstance(m.kernel_size, int):
                kk = m.kernel_size**2
            elif len(m.kernel_size) == 2:
                kk = np.prod(m.kernel_size)
            local_flop_adds += (kk - 1) * np.prod(output.size()) / d
            local_flop_mults += np.prod(output.size()) / d

            param_count += 0
            flop_adds += local_flop_adds
            flop_mults += local_flop_mults
            module_statistics[id(m)] = (m, [0, local_flop_adds, local_flop_mults])

        elif isinstance(m, nn.modules.pooling.AdaptiveAvgPool2d):
            assert input[0].shape[2] == input[0].shape[3], "The input is not square"

            # stencile/kernel size of the adaptive pooling
            stencil_size = (input[0].shape[2] + m.output_size - 1) // m.output_size
            local_flop_adds += (stencil_size * stencil_size - 1) * output.numel() / d
            local_flop_mults += output.numel() / d

            flop_adds += local_flop_adds
            flop_mults += local_flop_mults
            module_statistics[id(m)] = (m, [local_param_count, local_flop_adds, local_flop_mults])

        elif isinstance(m, InvertedResidual):
            # InvertedResidual is a high level module, which contains multiple submodules, including nn.Conv2d, ReLU_Module, etc.
            # In this code block, we only calculate the addition performed at the end of InvertedResidual layer, i.e., x += residual.

            # Not that submodules in InvertedResidual will be calculated in their own forward hooks, respectively.
            # For example, the conv2d layers in InvertedResidual will be calculated in the block -- 'if isinstance(m, nn.Conv2d):'.

            if m.has_residual:
                flop_adds += output.numel()  / d# For the addition at the end of InvertedResidual layer, i.e., x += residual
                module_statistics[id(m)] = (m, [0, output.numel() / d, 0])
            else:
                module_statistics[id(m)] = (m, [0,0,0])

        elif isinstance(m, SqueezeExcite):
            # SqueezeExcite is a high level module, which contains multiple submodules, including nn.Conv2d, ReLU_Module, etc.
            # All its submodules are properly registered with forward hooks.
            # In this code block, we only calculate the flops related to
            # 1) the global average pooling at the beginning of SE block;
            # 2) the multiplication at the end of SE block.

            # For the multiplication at the end of SE layer, i.e., x*y.expand_as(x)
            local_flop_mults += output.numel() / d

            # For the global average pooling at the beginning of SE block
            local_flop_mults += output.numel() / d
            local_flop_adds += (input[0].shape[2] * input[0].shape[3] - 1) * input[0].shape[1] * input[0].shape[0] / d

            flop_mults += local_flop_mults
            flop_adds += local_flop_adds
            module_statistics[id(m)] = (m, [local_param_count, local_flop_adds, local_flop_mults])

        elif isinstance(m, DepthwiseSeparableConv):
            # DepthwiseSeparableConv is used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
            # factor of 1.0. This is an alternative to having a IR with an optional first pointwise conv.
            # All its submodules are properly registered with forward hooks.
            # In this code block we only calculate the flops related to the additions at the end of DepthwiseSeparableConv block, i.e., x += residual
            if m.has_residual:
                flop_adds += output.numel() / d # For the residual addition
                module_statistics[id(m)] = (m, [0, output.numel() / d, 0])
            else:
                module_statistics[id(m)] = (m, [0, 0, 0])
        else:
            # Some high level modules are not counted as their submodles have already been registered with a forward hook.
            # Those high level modules are nn.Sequential, GenEfficientNet, etc.
            module_skipped.append(m)


    model.register_forward_hook(count_ops)


class ForwardSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        global alpha
        x_ternary = (x - x.mean())/x.std()
        ones = (x_ternary > alpha).type(torch.cuda.FloatTensor)
        neg_ones = -1 * (x_ternary < -alpha).type(torch.cuda.FloatTensor)
        x_ternary = ones + neg_ones
        multiplier = math.sqrt(2. / (x.shape[1] * x.shape[2] * x.shape[3]) * x_ternary.numel() / x_ternary.nonzero().size(0) )
        if args.amp:
            return (x_ternary.type(torch.cuda.HalfTensor), torch.tensor(multiplier).type(torch.cuda.HalfTensor))
        else:
            return (x_ternary.type(torch.cuda.FloatTensor), torch.tensor(multiplier).type(torch.cuda.FloatTensor))
    @staticmethod
    def backward(ctx, g):
        raise NotImplementedError("backward is only implemented at training time")

def forward_binarizable(self, x):
    w, M = self._get_weight('weight')
    conv = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
    conv = M * conv
    return ( conv  )

def _get_weight_binarizable(self, name):
    w = getattr(self, name)
    return ForwardSign.apply(w)

def Conv2d_binary_patch(m):
    m._get_weight = _get_weight_binarizable.__get__(m, m.__class__)
    m.forward = forward_binarizable.__get__(m, m.__class__)


def Model_binary_patch(model):
    def check_contain(k,L):
        res = [l in k for l in L]
        return(any(res))
    def disable_BatchNorm_affine(bn):
        bn.affine = False
        bn.weight = None
        bn.bias = None
    for k, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if check_contain(k, ['conv_stem', 'se.conv_reduce','se.conv_expand']): # does not convert the first conv layer and squeeze-excite
                continue
            elif m.kernel_size == (1,1): #  only the non-squeeze-excite conv with 1x1 kernel will be binarized
                Conv2d_binary_patch(m)


param_count, flop_mults, flop_adds = 0, 0, 0
module_skipped, module_statistics = [], {}


model = create_model(
    'MSUnet_CIFAR100',
    pretrained=False,
    num_classes=100,
    drop_rate=0.0,
    global_pool='avg',
    bn_tf=False,
    bn_momentum=None,
    bn_eps=None,
    checkpoint_path=args.initial_checkpoint)
Model_binary_patch(model)
model.apply(count_hook)
model.cuda()
if args.amp:
    model = amp.initialize(model, opt_level=args.opt_level)
model.eval()
with torch.no_grad():
    x = torch.randn([1, 3, 32, 32]).cuda()
    y = model(x)

flops = flop_mults+flop_adds
statistics_accumulations = {'Conv2d':[0,0,0],'BatchNorm2d':[0,0,0]}
for k in module_statistics:
    m = module_statistics[k][0].__class__
    m = str(m).split('.')[-1][:-2]
    n_params, n_adds, n_mults = module_statistics[k][1]
    print('Module {:25s} #params {:10.1f}({:10.5f}) \t #adds {:10.1f}({:10.5f}) \t #mults {:10.1f}({:10.5f})'.format(m, n_params, n_params/param_count, n_adds, n_adds/flops, n_mults, n_mults/flops))
    if m=='Conv2d':
        print(module_statistics[k][0])
    if m in statistics_accumulations:
        statistics_accumulations[m][0] += n_params
        statistics_accumulations[m][1] += n_adds
        statistics_accumulations[m][2] += n_mults
print(statistics_accumulations)



print('\n============================================')
flops = flop_mults+flop_adds
print('Flops {:.3E}/{:.3E} = {:.4f}, Params {:.3E}/{:.3E} = {:.4f}'.format(flops,10.49e9, flops/10.49e9, param_count,  36.5e6, param_count/36.5e6))
print('Overall score {:.5f}'.format(flops/10.49e9 + param_count/36.5e6))
print('=============================================')


