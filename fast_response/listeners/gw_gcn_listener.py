''' Script to automatically receive GCN alerts and get LIGO skymaps 
    to run realtime neutrino follow-up

    Author: Raamis Hussain
    Date:   Mar 27, 2019
'''

import gcn
import healpy as hp
import numpy as np
import os, subprocess
import lxml.etree

@gcn.handlers.include_notice_types(
    gcn.notice_types.LVC_PRELIMINARY,
    gcn.notice_types.LVC_INITIAL,
    gcn.notice_types.LVC_UPDATE)
def process_gcn(payload, root):

    analysis_path = os.environ.get('REALTIME_GW_PATH')
    if analysis_path is None:
        print('###########################################################################')
        print('CANNOT FIND ENVIRONMENT VARIABLE POINTING TO REALTIME GW ANALYSIS PACKAGE\n')
        print('put \'export /path/to/realtime_gw/release\' in your bashrc')
        print('###########################################################################')
        exit()

    # Respond only to 'test' events.
    # VERY IMPORTANT! Replace 'test' with
    # 'observation' to repond to real events.
    # if root.attrib['role']=='observation':
        ## Call everyone because it's a real event
        # call_command = analysis_path + '/make_call.py'
        # call_args = ['--raamis','--justin','--alex']
        # subprocess.call([call_command,call_args[0],'True',call_args[1],'True',call_args[2],'True'])

    # Read all of the VOEvent parameters from the "What" section.
    params = {elem.attrib['name']:
              elem.attrib['value']
              for elem in root.iterfind('.//Param')}

    # Read trigger time of event
    isot = lxml.etree.tostring(root.find('./WhereWhen/ObsDataLocation/ObservationLocation/'
                               'AstroCoords/Time/TimeInstant/ISOTime'))

    # Parse time string
    time_string = isot.split('>')[1].split('<')[0]
    date, time = time_string.split('T')
    trigger = date + ' ' + time

    print('\n')
    print('GW trigger time: %s \n' % trigger)

    skymap = params['skymap_fits']
    name = root.attrib['ivorn'].split('#')[1]

    #Print all parameters.
    for key, value in params.items():
        print(key, '=', value)

    command = analysis_path + '/run_gw_followup.py'
    args = ['--skymap', '--trigger','--name','--role']
    subprocess.call([command,args[0],skymap,args[1],trigger,args[2],name,args[3],root.attrib['role']])


print('Listening for GCNs...')
#gcn.listen(handler=process_gcn)

### FOR OFFLINE TESTING
payload = open('./S191216ap_update.xml', 'rb').read()
root = lxml.etree.fromstring(payload)
process_gcn(payload, root)
### For more test skymaps:
# https://gracedb.ligo.org/api/superevents/MS190403g/files/bayestar.fits.gz
