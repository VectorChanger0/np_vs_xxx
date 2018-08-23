import os
import random

cyclebin = os.path.join('..', 'tbd01')
def next_tbd_dir(dir0=cyclebin, maximum_int=100000):
    if not os.path.exists(dir0): os.makedirs(dir0)
    tmp1 = [x for x in os.listdir(dir0) if x[:3]=='tbd']
    exist_set = {x[3:] for x in tmp1}
    while True:
        tmp1 = str(random.randint(1,maximum_int))
        if tmp1 not in exist_set: break
    tbd_dir = os.path.join(dir0, 'tbd'+tmp1)
    os.mkdir(tbd_dir)
    return tbd_dir
