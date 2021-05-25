#!/usr/bin/env python

r'''Script to run followup to 
a realtime cascade alert

Author: Alex Pizzuto
May 2020'''

import numpy as np
import os, sys, argparse
from astropy.time import Time
from astropy.time import TimeDelta
import pandas as pd
import subprocess
import pickle

from fast_response.FastResponseAnalysis import FastResponseAnalysis
from . import utils

parser = argparse.ArgumentParser(description='Fast Response Analysis')
parser.add_argument('--skymap', type=str, default=None,
                    help='path to skymap')
parser.add_argument('--time', type=float, default=None,
                    help='Time of the alert event (mjd)')
parser.add_argument('--document', default=False, action='store_true')
parser.add_argument('--gcn_notice_num', default=0, type=int)
parser.add_argument('--alert_id', default=None,
                    type= lambda z:[ tuple(int(y) for y in x.split(':')) for x in z.split(',')],
                    help="list of events to exclude from this analysis. "
                    "such as HESE events that contributed to the trigger."
                    "Example --alert_id  127853:67093193,128290:6888376")
args = parser.parse_args()

cascade_time = Time(args.time, format='mjd')
year, month, day = cascade_time.iso.split('-')
day = day[:2]
casc_name = 'IceCube-Cascade_{}{}{}'.format(year[-2:], month, day)

all_results = {}
for delta_t in [1000., 2.*86400.]:
    start_time = cascade_time - (delta_t / 86400. / 2.)
    stop_time = cascade_time + (delta_t / 86400. / 2.)
    start = start_time.iso
    stop = stop_time.iso

    name = casc_name + ' {:.1e}_s'.format(delta_t)

    source = {}
    source['Name'] = name.replace('_', ' ')
    source['alert_event'] = True
    source['smear'] = False #CASCADES ARE ALREADY SENT AS PDFS NOT LLH
    source['alert_type'] = 'cascade'
    source['Skipped Events'] = args.alert_id

    f = FastResponseAnalysis(args.skymap, start, stop, **source)
    f.unblind_TS()
    f.plot_ontime()
    f.calc_pvalue()
    f.make_dNdE()
    f.plot_tsd()
    f.upper_limit()
    results = f.save_results()
    f.generate_report()
    if args.document:
        subprocess.call(['cp','-r',results['analysispath'],
        '/home/apizzuto/public_html/FastResponse/webpage/output/{}'.format(results['analysisid'])])
        utils.updateFastResponseWeb(results)
    all_results[delta_t] = results

all_results[1000.]['gcn_num'] = args.gcn_notice_num
utils.write_alert_gcn(all_results)

