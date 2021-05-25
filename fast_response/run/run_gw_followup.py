#!/usr/bin/env python

r''' This script will be called by gcn_listener.py and
     will start the neutrino follow up for a GW event. 
     Creates directory, gets data from livestream, does
     All sky scan, and makes report and GCN of results

    Author: Raamis Hussain
    Date:   Mar 27, 2019
'''


import os, argparse
import time as pythonTime
import numpy as np
import healpy as hp

from RealtimeFollowup         import RealtimeFollowup
from astropy.time             import Time
from ReportGenerator          import ReportGenerator
from skylab.datasets          import Datasets
from skylab.temporal_models   import TemporalModel,BoxProfile
from skylab.llh_models        import EnergyLLH
from skylab.ps_llh            import PointSourceLLH
from slack                    import slackbot
from numpy.lib.recfunctions   import append_fields


##################### CONFIGURE ARGUMENTS ########################
p = argparse.ArgumentParser(description="Calculates Sensitivity and Discovery"
                            " Potential Fluxes for Background Gravitational wave/Neutrino Coincidence study",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--skymap",required=True,type=str,help='URL to fits File for event localization')
p.add_argument("--trigger",required=True,type=str,help="Time of event trigger from GCN")
p.add_argument('--name', required=True,type=str,help='Name of event given by GraceID')
p.add_argument('--role', default='observation',type=str,help='Role of GW event. Either test or observation')
args = p.parse_args()
###################################################################
t0 = pythonTime.time()

trigger=args.trigger
date,time = trigger.split(' ')
skymap= args.skymap

#create directory for analysis. Place in appropriate directory for
#tests vs real observations
if args.role=='observation':
    dirname = os.environ.get('GW_OUTPUT')
    if dirname is None:
        dirname = os.environ.get('REALTIME_GW_PATH')
    dirname = dirname + '/%s' % args.name
else:
    dirname = os.environ.get('GW_MOCK_OUTPUT')
    if dirname is None:
        dirname = os.environ.get('REALTIME_GW_PATH')
    dirname = dirname + '/%s' % args.name
os.mkdir(dirname)

time_window = 500/86400. #500 seconds in days
trigger = Time(trigger,format='iso',scale='utc')
start = Time(trigger.mjd-time_window, format='mjd')
stop = Time(trigger.mjd+time_window, format='mjd')

### Configure Likelihood
dataset = Datasets['GFUOnline']
exp, mc, livetime, grl = dataset.livestream(start.mjd-6, start.mjd,
                         append=['IC86, 2017','IC86, 2018'],
                         floor=np.deg2rad(0.2))

print exp
print grl
sinDec_bins = dataset.sinDec_bins('IC86, 2017')
energy_bins = dataset.energy_bins('IC86, 2017')

llh_model = EnergyLLH(twodim_bins = [energy_bins, sinDec_bins],
                      allow_empty=True,bounds = [1.,4.],seed = 2.0,ncpu=20)
llh = PointSourceLLH(exp,mc,livetime,scramble=False,llh_model=llh_model,ncpu=20)
llh._warn_nsignal_max=False 
t1 = pythonTime.time()
print('Time to load initial data and set up likelihoods: %f' % (t1-t0))

latency = 60/86400. #add extra minute to make sure we have all events
exp_on, livetime_on, grl_on = dataset.livestream(start.mjd, stop.mjd+latency,load_mc=False,
                                      floor=np.deg2rad(0.2), wait_until_stop=True)

t2 = pythonTime.time()
print('Time to load on time events: %f' % (t2-t1))

llh.append_exp(exp_on,livetime_on)
llh.set_temporal_model(
                  TemporalModel(grl=np.concatenate([grl,grl_on]),poisson_llh=True,
                  signal=BoxProfile(start.mjd,stop.mjd),days=5))

# Initialize Realtime Follow Up object
rf = RealtimeFollowup(skymap,trigger.mjd,start.mjd,stop.mjd,llh=llh,dirname=dirname)

print('Starting All sky scan... \n')
tscan0 = pythonTime.time()
results, events = rf.realtime_scan()
tscan1 = pythonTime.time()
pvalue = rf.calc_pvalue(results['ts'])[0]
tpval1 = pythonTime.time()
scanTime = tscan1 - tscan0
pvalTime = tpval1 - tscan1

# Write GCN with results along with text file containing basic info
rf.write_results(dirname,results['ts'],results['ns'],results['gamma'],pvalue)


events = append_fields(events, names=['pvalue'],data=np.empty((1, events['ra'].size)),
                       usemask=False)

tperevent0 = pythonTime.time()
### compute per event pvalues below p=0.05
if pvalue<0.05:
  for i in range(events.size):
      res, ev = rf.realtime_scan(custom_events=events[i])
      p_best = rf.calc_pvalue(res['ts'])
      events['pvalue'][i] = p_best
else:
  pvals = rf.calc_pvalue(events['ts'])
  for i in range(events['ra'].size):
      events['pvalue'][i] = pvals[i]
tperevent1 = pythonTime.time()
tperevent = tperevent1 - tperevent0

gcn = rf.make_gcn(args.name,start.iso,stop.iso,events=events,pvalue=pvalue)

### make skymap with on-time events
tplots0 = pythonTime.time()
rf.plot_skymap(events)
rf.plot_skymap_zoom(events,results['ra'],results['dec'])
rf.make_decPDF(args.name.split('-')[0])
tplots1 = pythonTime.time()

# Generate report for event
rg = ReportGenerator(args.name,trigger.iso,start.iso,stop.iso,results,
                     events=events,dirname=dirname,pvalue=pvalue)
rg.generate_gw_report()
rg.make_pdf()

#send slack message to chosen channel
# url = "https://hooks.slack.com/services/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" # For Josh test channel
# if args.role=='observation':
#    roc  = 'https://hooks.slack.com/services/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' # For ROC
#    link = 'https://icecube.wisc.edu/~rhussain/O3_followups/'+args.name+'/'+date+'_'+args.name+'_report.pdf'
#    realtime = 'https://hooks.slack.com/services/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'
#    s1 = slackbot('roc', 'gw_alert', roc)
#    s2 = slackbot('realtime','GW Alert',realtime)
#    s1.send_message("Neutrino Follow up for GW Event %s is complete.\nReport: <%s|link>" % (args.name,link), "gw")
#    s2.send_message("Neutrino Follow up for GW Event %s is complete.\nReport: <%s|link>" % (args.name,link), "gw")

# Check if any GFU events are available after the on-time window to
# make sure we have considered all events in the window
t3 = pythonTime.time()
new_stop = (t3-t2)/86400. + stop.mjd
exp, livetime, grl = dataset.livestream(stop.mjd,new_stop,load_mc=False,
                                      floor=np.deg2rad(0.2))

gfu_status = open(dirname + '/gfu_status.txt','w+')
gfu_status.write('Check below to see if there are GFU events after the end of the on-time window.')
gfu_status.write('If there are no events, this could indicate a problem with GFU latency.\n')
gfu_status.write('-----------------------------------------------------------------------------\n\n')
gfu_status.write('Number of GFU events between on-time window and end of this analysis: %s \n' % exp['time'].size)
if exp['time'].size==0:
    gfu_status.write('#######################################################################')
    gfu_status.write('### WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ###')
    gfu_status.write('#######################################################################')


print('Time taken for analysis and reports: %f' % (t3-t2))
print('Total time taken: %f' % (t3-t0))
rf.write_results(dirname,results['ts'],results['ns'],results['gamma'],pvalue,latency=[(t3-t2),(t3-t0)])
print('Follow up analysis complete')
print('Report Created in directory: %s' % dirname)
lat_file = open(dirname + '/latency.txt', 'w')
lines = ['InitializeLLH {}\n'.format(t1 - t0),
            'LoadOnTimeData {}\n'.format(t2 - t1),
            'RunScan {}\n'.format(scanTime),
            'AllSkyPValue {}\n'.format(pvalTime),
            'SeparatePValues {}\n'.format(tperevent),
            'MakeReport {}\n'.format(t3 - tplots1),
            'MakePlots {}\n'.format(tplots1 - tplots0),
            'Overall {}\n'.format(t3 - t0)]
lat_file.writelines(lines)
lat_file.close()


