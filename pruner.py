import torch
import numpy as np



class ParameterMasker(object):
    """
    Adapted from Neural network distiller https://github.com/NervanaSystems/distiller
    A ParameterMasker can mask a parameter tensor or a gradients tensor.

    It is used when pruning DNN weights.
    """
    def __init__(self, param_name):
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.is_regularization_mask = False
        self.use_double_copies = False
        self.mask_on_forward_only = False
        self.unmasked_copy = None
        self.backward_hook_handle = None

    def apply_mask(self, parameter):
        """Apply a mask on the weights tensor (parameter)."""
        if self.mask is None:
            return
        if self.use_double_copies:
            self.unmasked_copy = parameter.clone().detach()
        self.mask_tensor(parameter)
        if self.is_regularization_mask:
            self.mask = None
        return parameter

    def mask_tensor(self, tensor):
        if self.mask is not None:
            tensor.data.mul_(self.mask)

    def mask_gradient(self, gradient):
        if self.mask is not None:
            return gradient.mul(self.mask)

    def revert_weights(self, parameter):
        if not self.use_double_copies or self.unmasked_copy is None:
            return
        parameter.data.copy_(self.unmasked_copy)
        self.unmasked_copy = None



class Pruner(object):
    '''
    Adapted from Neural network distiller https://github.com/NervanaSystems/distiller
    :param object:
    :return:
    The mask should not be used on Ternary weights
    '''
    def __init__(self, model, start_epoch, end_epoch, sparsity_dict):
        self.model = model
        self.sparsity_dict = sparsity_dict
        self.mask_dict = self._get_mask_dict()
        self.start_epoch, self.end_epoch = start_epoch, end_epoch

    def _get_mask_dict(self):

        mask_dict = {}
        for k in self.sparsity_dict:
            mask_dict[k] = ParameterMasker(k)
        return mask_dict


    def _get_current_sparsity(self, epoch):
        if epoch <= self.end_epoch:
            r = (epoch - self.start_epoch)/(self.end_epoch - self.start_epoch)
        else:
            r = 1
        s = {w:self.sparsity_dict[w] - self.sparsity_dict[w]*(1-r)**3 for w in self.sparsity_dict}
        #print('Current Sparsity is {}'.format(s))
        return s

    def _update_all_mask(self, epoch):
        current_sparsity_dict = self._get_current_sparsity(epoch)
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                s = current_sparsity_dict[k]
                self._mask_to_target_sparsity(self.mask_dict[k], w.data, s)


    def _mask_to_target_sparsity(self, mask, weight, sparsity):
        bottomk, _ = torch.topk(weight.abs().view(-1), int(sparsity*weight.numel()), largest=False, sorted=True)
        if len(bottomk) > 0:
            threshold = bottomk.data[-1]
        else:
            threshold = 0
        if sparsity > 0:
            mask.mask = self._threshold_mask(weight, threshold).requires_grad_(False)
        else:
            mask.mask = torch.ones_like(weight).requires_grad_(False)
        # if mask.mask is None: # initialize mask
        #     mask.mask = torch.ones_like(weight).requires_grad_(False)


    def _threshold_mask(self, weight, threshold):
        return torch.gt(torch.abs(weight), threshold).type(weight.type())

    def on_epoch_begin(self,epoch):
        self._update_all_mask(epoch)

    def on_minibatch_begin(self):
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                self.mask_dict[k].apply_mask(w)

    def on_minibatch_end(self):
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                self.mask_dict[k].apply_mask(w)

    def print_statistics(self):
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                #self.mask_dict[k].apply_mask(w)
                sparsity = (w.numel() - w.nonzero().size(0)) / w.numel()
                print('Pruner: {} with sparsity {}'.format(k, sparsity))

def print_sparsity_statistics(model, print=True):
    weight_list = ['blocks.2.0.se.conv_reduce.weight',
                   'blocks.2.0.se.conv_expand.weight',
                   'blocks.2.1.se.conv_reduce.weight',
                   'blocks.2.1.se.conv_expand.weight',
                   'blocks.3.0.se.conv_reduce.weight',
                   'blocks.3.0.se.conv_expand.weight',
                   'blocks.3.1.se.conv_reduce.weight',
                   'blocks.3.1.se.conv_expand.weight',
                   'blocks.1.0.conv_dw.weight',
                   'blocks.2.0.conv_dw.0.weight',
                   'blocks.2.0.conv_dw.1.weight',
                   'blocks.2.0.conv_dw.2.weight',
                   'blocks.2.1.conv_dw.0.weight',
                   'blocks.2.1.conv_dw.1.weight',
                   'blocks.3.0.conv_dw.0.weight',
                   'blocks.3.0.conv_dw.1.weight',
                   'blocks.3.0.conv_dw.2.weight',
                   'blocks.3.1.conv_dw.0.weight',
                   'blocks.3.1.conv_dw.1.weight',
                   'classifier.weight'
                   ]
    sparsity_ = []
    for k, w in model.named_parameters():
        if k in weight_list:
            # self.mask_dict[k].apply_mask(w)
            sparsity = (w.numel() - w.nonzero().size(0)) / w.numel()
            if print:
                print('{} with sparsity {}'.format(k, sparsity))
            sparsity_.append(sparsity)
    return np.mean(sparsity_)

def cal_current_sparsity(epoch, total_epoch, sparsity):
    r = epoch / total_epoch
    s = sparsity - sparsity * (1 - r) ** 3
    return(s)




class Pruner_mixed(object):
    def __init__(self, model, start_epoch, end_epoch, pruner_type):
        self.model = model
        #self.mask_dict, self.sparsity_dict = self._get_mask_dict()
        self.start_epoch, self.end_epoch = start_epoch, end_epoch

        self.sparsity_se, self.sparsity_dw, self.sparsity_classifier = self.get_sparsity_dicts(0.9, 0.5, 0.9)
        if pruner_type in ['V1','v1']:
            self.pruner_list = [Pruner(model, start_epoch, end_epoch, self.sparsity_se)]
        elif pruner_type in ['V2', 'v2']:
            self.pruner_list = [Pruner(model, start_epoch, end_epoch, self.sparsity_se),
                               Pruner(model, start_epoch, end_epoch, self.sparsity_dw)]
        elif pruner_type in ['V3', 'v3']:
            self.pruner_list = [Pruner(model, start_epoch, end_epoch, self.sparsity_se),
                               Pruner(model, start_epoch, end_epoch, self.sparsity_dw),
                                Pruner(model, start_epoch,end_epoch, self.sparsity_classifier)]

    def on_epoch_begin(self,epoch):
        for pruner in self.pruner_list:
            pruner.on_epoch_begin(epoch)

    def print_statistics(self):
        for pruner in self.pruner_list:
            pruner.print_statistics()

    def on_minibatch_begin(self):
        for pruner in self.pruner_list:
            pruner.on_minibatch_begin()

    def on_minibatch_end(self):
        for pruner in self.pruner_list:
            pruner.on_minibatch_end()

    def get_sparsity_dicts(self, s_se=0.9, s_dw=0.5, s_fc=0.9):
        sparsity_se = {'blocks.2.0.se.conv_reduce.weight': s_se,
                       'blocks.2.0.se.conv_expand.weight': s_se,
                       'blocks.2.1.se.conv_reduce.weight': s_se,
                       'blocks.2.1.se.conv_expand.weight': s_se,
                       'blocks.3.0.se.conv_reduce.weight': s_se,
                       'blocks.3.0.se.conv_expand.weight': s_se,
                       'blocks.3.1.se.conv_reduce.weight': s_se,
                       'blocks.3.1.se.conv_expand.weight': s_se
                       }
        sparsity_dw = {'blocks.2.0.conv_dw.0.weight': s_dw,
                          'blocks.2.0.conv_dw.1.weight': s_dw,
                          'blocks.2.0.conv_dw.2.weight': s_dw,
                          'blocks.2.1.conv_dw.0.weight': s_dw,
                          'blocks.2.1.conv_dw.1.weight': s_dw,
                          'blocks.3.0.conv_dw.0.weight': s_dw,
                          'blocks.3.0.conv_dw.1.weight': s_dw,
                          'blocks.3.0.conv_dw.2.weight': s_dw,
                          'blocks.3.1.conv_dw.0.weight': s_dw,
                          'blocks.3.1.conv_dw.1.weight': s_dw
                          }
        sparsity_classifier = {'classifier.weight': s_fc}
        return(sparsity_se, sparsity_dw, sparsity_classifier)
