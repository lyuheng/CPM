import numpy as np
import torch
import math

import shutil
import yaml
from easydict import EasyDict as edict

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0.
		self.avg = 0.
		self.sum = 0.
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint'):
    
    torch.save(state, filename + '_latest.pkl')
    if is_best:
        shutil.copyfile(filename + '_latest.pkl', filename + '_best.pkl')


def Config(filename):

    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser


def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):

    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr
