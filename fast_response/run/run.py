r'''Script to initiate fast reponse
    followups. Arguments can be passed
    at the beginning of execution or as 
    user I/O 

    Author: Alex Pizzuto
    Date: April 9, 2019
    '''

import os.path,sys,argparse
import subprocess
import warnings
import utils
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Fast Response Analysis')
parser.add_argument('--name', type=str,default=None,
                    help='Name of the source')
parser.add_argument('--loc', type=str, default=None,
                    help='Location of source, either as RA, Dec in degrees or path to skymap')
parser.add_argument('--start', type=str,default=None,
                    help="Start of the onsource time window relative "
                    "to the trigger in seconds")
parser.add_argument('--stop', type=str,default=None,
                    help="End of the onsource time window relative "
                    "to the trigger in seconds")
parser.add_argument('--extension', type=float, default=None,
                    help="Source extension in degrees")
parser.add_argument('--skip-events', default=None,
                    type= lambda z:[ tuple(int(y) for y in x.split(':')) for x in z.split(',')],
                    help="list of events to exclude from this analysis. "
                    "such as HESE events that contributed to the trigger."
                    "Example --skip-events  127853:67093193,128290:6888376")
parser.add_argument('--ntrials', default=1000, type=int,
                    help="Number of background trials to perform")
parser.add_argument('--document', default=False, action='store_true')
parser.add_argument('--nodiag', default=False, action='store_true')
parser.add_argument('--alert_event', default=False, action='store_true')
args = parser.parse_args()

subprocess.call(['clear'])
try:
    import pyfiglet
    message = '*'*80
    message += '\n' + str(pyfiglet.figlet_format("Fast Response Followup")) + '\n'
    message += '*'*80
    print(message)
except:
    message = '*'*80
    message += '\n' + 'Fast Response Followup' + '\n'
    message += '*'*80
    print(message)
#print(args.skip_events)
#User i/o for reuquired variables that were missed
final_args = {}
for argument, name, note in [(args.name, 'Name', 'Enter name of source'), 
                            (args.loc, 'Location', 'Enter as ra,dec in degrees or path to skymap'), 
                            (args.start, 'Start Time', 'Enter in iso'),
                            (args.stop, 'Stop Time', 'Enter in iso'), 
                            (args.skip_events, 'Skipped Events', 'Hit enter if no events to skip'),
                            (args.extension, 'Extension', 'Hit enter if no extension')]:
    if argument is None:
        if argument in [args.skip_events, args.extension]:
            continue
        print("\nUser did not enter a {}, please enter it now".format(name))
        new_arg = raw_input("{}:   ".format(note))
        if new_arg == '':
            new_arg = None
        final_args[name] = new_arg
    else:
        final_args[name] = argument
print('')   
import logging as log
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
from fast_response.FastResponseAnalysis import FastResponseAnalysis

log.basicConfig(level=log.ERROR)
source = final_args
source['alert_event'] = args.alert_event
####################### INITIALIZE FAST RESPONSE OBJECT #######################
f = FastResponseAnalysis(source['Location'], source['Start Time'], source['Stop Time'], **source)
# Point source, gw, etc. handling done in analysis object instantiation

f.unblind_TS()
f.plot_ontime()
if not args.alert_event:
    f.ns_scan()
f.calc_pvalue(ntrials = args.ntrials)
if not args.nodiag:
    f.make_dNdE()
    f.plot_tsd()
    f.upper_limit()
    #f.ns_scan()
results = f.save_results()
f.generate_report()
if args.document:
    subprocess.call(['cp','-r',results['analysispath'],
        '/home/apizzuto/public_html/FastResponse/webpage/output/{}'.format(results['analysisid'])])
    utils.updateFastResponseWeb(results)
#f.display_results()
#f.save()

