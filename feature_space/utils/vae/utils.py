import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmUp(object):
    "Returns the value of the anneling factor to be used"

    def __init__(self, epochs=100, value=1.0):
        self.epoch = 0
        self.max_epoch = epochs
        self.value = value

    def get(self):

        if self.epoch >= self.max_epoch:
            return self.value
        else:
            return self.value*(float(self.epoch)/self.max_epoch)

    def update(self):
        self.epoch += 1



class WarmUpStep(object):
    "Returns the value of the anneling factor to be used"

    def __init__(self, epochs=100, value=1.0):
        self.epoch = 0
        self.max_epoch = epochs
        self.value = value

    def get(self):

        if self.epoch >= self.max_epoch:
            return self.value
        else:
            return 0.0

    def update(self):
        self.epoch += 1

def sym_KLD_gaussian(p_mu,p_logvar,q_mu,q_logvar):

    p_var = torch.exp(p_logvar)
    q_var = torch.exp(q_logvar)

    mu_diff = torch.pow(p_mu-q_mu,2)
    first_term = 0.5*(mu_diff+q_var)/p_var
    second_term = 0.5*(mu_diff+p_var)/q_var
    return torch.sum(first_term+second_term-1,dim=-1)