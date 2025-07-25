import sys
import os
import time

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')
        self.path = path

    def cprint(self, text):
        print(text)
        self.f.write(f'{text}\n')
        self.f.flush()

    def close(self):
        self.f.close()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count