#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to sample with different temperatures from a trained model.
Need to turn off 'ask' when initializing model
"""

import os

sample = 36209
#temps = [0.5, 1., 1.5]
temps = [1.25]

name = '9_uniprot_train_lstm'
sepoch = 29
maxlen = 500

pwd = '/root/autodl-tmp/LSTM_peptides/'
modfile = pwd + name + '/checkpoint/model_epoch_%i.hdf5' % sepoch

for t in temps:
    print("\nSampling %i sequences at %.1f temperature..." % (sample, t))
    cmd = "python %sLSTM_peptides_sample.py --train False --modfile %s " \
          "--temp %.1f --sample %i --maxlen %i --name mjs" % (pwd, modfile, t, sample, maxlen)
    os.system(cmd)
