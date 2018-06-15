#!/usr/bin/env python
# encoding: utf-8

import os
cmd = os.path.join(os.getcwd(), "demo_training.py")
for i in range(10):
    os.system('{} {}'.format('python3', cmd))
