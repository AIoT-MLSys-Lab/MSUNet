from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler, StepLRScheduler2


def create_scheduler(args, optimizer):
    num_epochs = args.epochs
    lr_scheduler = None
    #FIXME expose cycle parms of the scheduler config to arguments
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=args.min_lr,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=args.cycle_limit,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'step' and isinstance(args.decay_epochs, int):
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
        )
    elif args.sched == 'step' and isinstance(args.decay_epochs, list):
        lr_scheduler = StepLRScheduler2(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
        )
    else:
        raise NotImplementedError('We do not have the scheduler you specified')
    return lr_scheduler, num_epochs
