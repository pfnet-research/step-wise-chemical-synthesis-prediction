import json
import numpy as cp

import logging
import argparse

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='result_all/inference1_final.txt')
args = parser.parse_args()


a = []
with open(args.path, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        a.append(line.strip())

correct = 0
for i in a:
    d = json.loads(i)
    label = d['raw']['label'].split(';')

    # number of corrected step
    cor = 0

    # check whether step number is correct
    if cp.where(cp.append(cp.asarray(d['stops']), 1.0) == 1.0)[0][0] + 1 != len(label):
        continue
    # step-wise top1 pair with action
    for j in range(len(label)):
        p = cp.asarray(d['pairs'])[j] + 1
        # e.g. [13-15-1.0] in real actions
        if str(int(p[0])) + '-' + str(int(p[1])) + '-' + str(d['actions'][j]) in label or str(int(p[1])) + '-' + str(
                int(p[0])) + '-' + str(d['actions'][j]) in label:
            cor += 1

    if cor == len(label):
        correct += 1

print(correct / len(a))
