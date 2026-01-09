from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)
        self.reset()

    def reset(self):
        self.sum.clear()
        self.count.clear()

    def update(self, output):
        for k, v in output.items():
            self.sum[k] += v.sum().item()
            self.count[k] += v.shape[0]
            self.avg[k] = self.sum[k] / self.count[k]

    @property
    def average(self):
        return self.avg

    def __repr__(self):
        messege = ""
        for k in self.avg.keys():
            messege += "\t" + k + f"={self.avg[k]:.4f}"
        return messege
