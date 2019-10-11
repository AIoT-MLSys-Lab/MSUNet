import argparse
import time
import yaml



from apex import amp


from timm.data import resolve_data_config
from timm.models import create_model
from timm.utils import *


import torch
import torch.nn as nn
import torchvision.utils
import torch.nn.functional as F

from timm.data import create_loader_CIFAR100



torch.backends.cudnn.benchmark = True



# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='Training')
# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--binarizable', type=str, default='T', help='Using binary (B) or Tenary (T)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--alpha', default=None, type=float, help='Hard alpha-binary threshold for Ternary')

parser.add_argument('--amp', action='store_true', default=False, help='use NVIDIA amp for mixed precision training')
parser.add_argument('--sync-bn', action='store_true', help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default='O1', help='Apex opt-level, "01"(conservative), "03"(Pure FP16)')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)


    return args, args_text

def get_alpha(epoch, args):
    if args.binarizable == 'T':
        print('Using ternary 1x1 conv weights.')
        if args.alpha is not None:
            alpha = args.alpha # use predefined threshold
        else:
            if epoch < 20:  # The first 20 epochs are used as warm up
                alpha = 0
            elif epoch < 400:
                r = (epoch - 20) / (400 - 20)
                alpha = r * 0.67449
            else:
                alpha = 0.67749  # scipy.stats.norm.ppf(0.75)
    elif args.binarizable == 'B':
        print('Using binary 1x1 conv weights.')
        alpha = 0.0
    else:
        raise ValueError('option --binarizable is incorrect')
    return alpha


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


args = None

def main():
    global args

    setup_default_logging()
    args, args_text = _parse_args()

    global alpha
    alpha = args.alpha
    assert alpha == 0.67749, 'Make sure alpha = 0.67749'

    args.device = 'cuda:0'
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        global_pool=args.gp,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        checkpoint_path=args.initial_checkpoint)

    assert args.binarizable == 'T', 'Make sure --binarizable T'
    if args.binarizable:
        Model_binary_patch(model)

    model.cuda()

    use_amp = False
    if has_apex and args.amp:
        print('Using amp with --opt-level {}.'.format(args.opt_level))
        model = amp.initialize(model, opt_level=args.opt_level)
        use_amp = True
    else:
        print('Do NOT use amp.')


    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    dataset_eval = torchvision.datasets.CIFAR100(root='~/Downloads/CIFAR100', train=False, download=True)
    loader_eval = create_loader_CIFAR100(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=4 * args.batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=None,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=False,
    )

    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    eval_metrics = validate(model, loader_eval, validate_loss_fn, args)





def validate(model, loader, loss_fn, args, log_suffix='', tensorboard_writer=None, epoch=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data
            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 10 == 0):
                global alpha
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f}) '
                    'Alpha: {alpha} '.format(
                        log_name, batch_idx, last_idx,
                        batch_time=batch_time_m, loss=losses_m,
                        top1=prec1_m, top5=prec5_m, alpha=alpha))

    logging.info('The average top-1 test accuracy is {top1.avg:>7.4f}'.format(top1=prec1_m))
    metrics = OrderedDict([('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg)])

    return metrics

alpha = 0.67749
if __name__ == '__main__':
    main()
