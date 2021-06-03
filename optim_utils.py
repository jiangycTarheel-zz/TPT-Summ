# Adafactor optimizer adapted from: https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

import math
from numpy import sqrt

import torch
import torch.optim
from torch.optim.lr_scheduler import LambdaLR


def get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases as 1/sqrt(current_step) after
    being constant during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return max(
            0.0, 1.0 / max(1., sqrt(float(max(current_step, num_warmup_steps))))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if true, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if true, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None,  weight_decay=0.0, scale_parameter=True,
                 relative_step=True, warmup_init=False, fix_lr_step=30000):
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
                        beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super(Adafactor, self).__init__(params, defaults)
        self.fix_lr_step = fix_lr_step

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            #min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            #rel_step_sz = min(min_step, 1.0/math.sqrt(param_state['step']))
            if param_state['step'] <= self.fix_lr_step:
                rel_step_sz = 1e-2
            else:
                rel_step_sz = 1.0/math.sqrt(param_state['step'] - (self.fix_lr_step - 10000.))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col, output):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0

                state['step'] += 1
                state['RMS'] = self._rms(p.data)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = (grad**2) + group['eps'][0]
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    exp_avg_sq_row.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-1))
                    exp_avg_sq_col.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-2))
                    # Approximation of exponential moving average of square of gradient
                    self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(1.0 - beta2t, update)
                    torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                update.div_(max(1.0, self._rms(update) / group['clip_threshold']))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(1 - group['beta1'], update)
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * lr, p.data)

                p.data.add_(-update)

        return loss