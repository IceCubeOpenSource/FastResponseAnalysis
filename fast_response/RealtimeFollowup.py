r''' GW Followup class. Contains object to run follow up
    analysis for GW events. Also has  methods to 
    generate reports and GCN notices

    Author: Raamis Hussain
    Date: Mar 10th, 2019
    '''

import numpy  as np
import healpy as hp
import time, datetime, os
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import meander
import matplotlib.lines as mlines
import logging 
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 

from matplotlib               import cm
from skylab.priors            import SpatialPrior
from skylab.ps_injector       import PointSourceInjector,PriorInjector
from astropy.time             import Time
from scipy.special            import erfinv
from numpy.lib.recfunctions   import append_fields

class RealtimeFollowup(object):
    r''' Object to do realtime followup analyses of 
        astrophysical transients with arbitrary event
        localization
    '''
    _angScale = 2.145966 #Scale neutrino angular error from 39%->90%
                         #containment of a 2d gaussian.

    def __init__(self,skymap,trigger,tstart,tstop,**kwargs):
        r'''Constructor

        Parameters:
        -----------
        skymap: str
            Path to local fits file or URL to fits file to be
            downloaded
        trigger: str
            Time of event in UTC. Must have the form:
            '2019-03-10 12:00:00' 
        tstart: 
        tstop

        '''
        self.skymap_url = skymap
        self.skymap = hp.read_map(skymap)
        if hp.pixelfunc.get_nside(self.skymap)!=512:
            self.skymap = hp.pixelfunc.ud_grade(self.skymap,512,power=-2)

        self.nside = hp.pixelfunc.get_nside(self.skymap)
        self.ipix_90 = self.ipixs_in_percentage(self.skymap,0.9)

        self.dirname = kwargs.pop('dirname','/home/rhussain/icecube/dump/')
        self.analysis_path = os.environ.get('REALTIME_GW_PATH')
        self.llh = kwargs.pop('llh',None)
        self.trigger = trigger
        self.start = tstart
        self.time_window = trigger-tstart
        self.stop = tstop
        if self.llh is None:
            raise ValueError('Need LLH object to do analysis')

        self.pre_ts_array = None
        self.max_ts = None

    def realtime_scan(self,custom_events=None):
        r''' Run All sky scan using event localization as 
        spatial prior.

        Returns:
        --------
        ts: array
            array of ts values of 
        ''' 
        ### Set up spatial prior to be used in scan
        spatial_prior = SpatialPrior(self.skymap,allow_neg=True)
        pixels = np.arange(len(self.skymap))

        val = self.llh.scan(0.0,0.0, scramble = False,spatial_prior=spatial_prior,
                            time_mask = [self.time_window,self.trigger],
                            pixel_scan=[self.nside,3.0],custom_events=custom_events)

        exp_theta = 0.5*np.pi - self.llh.exp['dec']
        exp_phi   = self.llh.exp['ra']
        exp_pix   = hp.ang2pix(self.nside,exp_theta,exp_phi)
        overlap   = np.isin(exp_pix,self.ipix_90)

        t_mask=(self.llh.exp['time']<=self.stop)&(self.llh.exp['time']>=self.start)
        events = self.llh.exp[t_mask]

        # add field to see if neutrino is within 90% GW contour
        if custom_events is None:
            events = append_fields(events, names=['in_contour','ts','ns','gamma','B'],
                                  data=np.empty((5, events['ra'].size)),
                                  usemask=False)

            for i in range(events['ra'].size):
                events['in_contour'][i]=overlap[i]

            for i in range(events['ra'].size):
                events['B'][i] = self.llh.llh_model.background(events[i])

        if val['TS'].size==0:
            return (-1*np.inf,0,2.0,None)
        else:
            ts=val['TS_spatial_prior_0'].max()
            maxLoc = np.argmax(val['TS_spatial_prior_0'])
            ns=val['nsignal'][maxLoc]
            gamma=val['gamma'][maxLoc]
            ra = val['ra'][maxLoc]
            dec = val['dec'][maxLoc]
        

        val_pix = hp.ang2pix(self.nside,np.pi/2.-val['dec'],val['ra'])
        for i in range(events['ra'].size):
            idx, = np.where(val_pix==exp_pix[i])
            events['ts'][i] = val['TS_spatial_prior_0'][idx[0]]
            events['ns'][i] = val['nsignal'][idx[0]]
            events['gamma'][i] = val['gamma'][idx[0]]

        results = dict([('ts',ts),('ns',ns),('gamma',gamma),('ra',ra),('dec',dec)])
        return (results,events)

    def inject_scan(self,ra,dec,ns,poisson=True):
        r''' Run All sky scan using event localization as 
        spatial prior, while also injecting events according
        to event localization

        Parameters:
        -----------
        ns: float
            Number of signal events to inject
        poisson: bool
            Will poisson fluctuate number of signal events
            to be injected
        Returns:
        --------
        ts: array
            array of ts values of 
        ''' 
        ### Set up spatial prior to be used in scan
        spatial_prior = SpatialPrior(self.skymap,allow_neg=True)
        pixels = np.arange(len(self.skymap))

        ## Perform all sky scan
        inj = PointSourceInjector(gamma=2,E0=1000.)
        inj.fill(dec,self.llh.exp, self.llh.mc, self.llh.livetime,
                 temporal_model=self.llh.temporal_model) 
        ni, sample = inj.sample(ra,ns,poisson=poisson)
        print('injected neutrino at:')
        print(np.rad2deg(sample['ra']),np.rad2deg(sample['dec']))

        val = self.llh.scan(0.0,0.0, scramble = False,spatial_prior=spatial_prior,
                            time_mask = [self.time_window,self.trigger],inject=sample,
                            pixel_scan=[self.nside,3.0])

        exp = self.llh.inject_events(self.llh.exp,sample)
        exp_theta = 0.5*np.pi - exp['dec']
        exp_phi   = exp['ra']
        exp_pix   = hp.ang2pix(self.nside,exp_theta,exp_phi)
        overlap   = np.isin(exp_pix,self.ipix_90)

        t_mask=(exp['time']<=self.stop)&(exp['time']>=self.start)
        events = exp[t_mask]

        # add field to see if neutrino is within 90% GW contour
        events = append_fields(events, names=['in_contour','ts','ns','gamma','B'],
                              data=np.empty((5, events['ra'].size)),
                              usemask=False)

        for i in range(events['ra'].size):
            events['in_contour'][i]=overlap[i]

        for i in range(events['ra'].size):
            events['B'][i] = self.llh.llh_model.background(events[i])

        if val['TS'].size==0:
            return (0,0,2.0,None)
        else:
            ts=val['TS_spatial_prior_0'].max()
            maxLoc = np.argmax(val['TS_spatial_prior_0'])
            ns=val['nsignal'][maxLoc]
            gamma=val['gamma'][maxLoc]
            ra = val['ra'][maxLoc]
            dec = val['dec'][maxLoc]
        
        val_pix = hp.ang2pix(self.nside,np.pi/2.-val['dec'],val['ra'])
        for i in range(events['ra'].size):
            idx, = np.where(val_pix==exp_pix[i])
            events['ts'][i] = val['TS_spatial_prior_0'][idx[0]]
            events['ns'][i] = val['nsignal'][idx[0]]
            events['gamma'][i] = val['gamma'][idx[0]]

        results = dict([('ts',ts),('ns',ns),('gamma',gamma),('ra',ra),('dec',dec)])
        return (results,events)

    def calc_pvalue(self,ts,month=None):
        r''' Calculate pvalue by applying spatial prior to
        a pre-computed map of TS values

        Parameters:
        -----------
        ts: float
            Test statistic of result
        month: int, optional
            Int referring to month of desired precomputed array to use. 

        Returns:
        --------
        pvalue: float
            P-value corresponding to the given ts
        '''
        ts = np.atleast_1d(ts)
        if month is None:
            month = datetime.datetime.utcnow().month

        if self.pre_ts_array is None:
            self.pre_ts_array = np.load('/data/user/rhussain/ligo_skymaps/ts_map_%02d.npy' % month,
                                        allow_pickle=True)

        # Create spatial prior weighting
        if self.max_ts is None:
            max_ts = []
            ts_norm = np.log(np.amax(self.skymap))
            for i in range(self.pre_ts_array.size):
                # If a particular scramble in the pre-computed ts_array is empty,
                #that means that sky scan had no events in the sky, so max_ts=0
                if self.pre_ts_array[i]['ts'].size==0:
                    max_ts.append(-1*np.inf)
                else:
                    theta, ra = hp.pix2ang(512,self.pre_ts_array[i]['pixel'])
                    dec = np.pi/2. - theta
                    interp = hp.get_interp_val(self.skymap,theta,ra)
                    interp[interp<0] = 0.
                    ts_prior = self.pre_ts_array[i]['ts'] + 2*(np.log(interp) - ts_norm)
                    max_ts.append(ts_prior.max())

            max_ts = np.array(max_ts)
            self.max_ts = max_ts
        
            pvalue = []
            for i in range(ts.size):
                pvalue.append(max_ts[max_ts>=ts[i]].size/float(max_ts.size))

            return np.array(pvalue)

        pvalue = []
        for i in range(ts.size):
            pvalue.append(self.max_ts[self.max_ts>=ts[i]].size/float(self.max_ts.size))

        return np.array(pvalue)

    def ps_sens_range(self):
        r''' Compute minimum and maximum sensitivities within
        the declination range of the 90% contour of a given skymap

        Returns:
        --------
        low: float
            lowest sensitivity within dec range
        high: floaot
            highest sensitivity wihtin dec range
        '''
        dec_range = np.linspace(-85,85,35)
        sens = [1.15,1.06,.997,.917,.867,.802,.745,.662,.629,.573,.481,.403,
                .332,.250,.183,.101,.035,.0286,.0311,.0341,.0361,.0394,.0418,
                .0439,.0459,.0499,.0520,.0553,.0567,.0632,.0679,.0732,.0788,.083,.0866]

        src_theta, src_phi = hp.pix2ang(self.nside,self.ipix_90)
        src_dec = np.pi/2. - src_theta
        src_dec = np.unique(src_dec)

        sens = np.interp(np.degrees(src_dec),dec_range,sens)
        low = sens.min()
        high = sens.max()

        return low,high
    
    def plot_skymap(self,events):
        r''' Make skymap with event localization and all
        neutrino events on the sky withing the given time window

        Parameters:
        -----------
        events: ndarray
            array of events to plot on the sky. Must have 'ra',
            'dec', and 'sigma' fields in array
        '''

        # Set color map and plot skymap
        cmap = cm.YlOrRd
        cmap.set_under("w")
        hp.mollview(self.skymap,cbar=True,coord='C',unit=r'Probability',
                    rot=180,cmap=cmap)
        hp.graticule()

        theta=np.pi/2 - events['dec']
        phi = events['ra']
        hp.projscatter(theta,phi,c='b',marker='x',label='IC Event (90\%)')

        # Scale sigma for 2d gaussian to get 90% containment
        sigma_90 = events['sigma']*self._angScale

        ## plot events on sky with error contours
        for i in range(events['ra'].size):
            my_contour = self.contour(events['ra'][i],events['dec'][i],sigma_90[i],self.nside)
            hp.projplot(my_contour[0], my_contour[1], linewidth=2., color="gray", linestyle="solid",coord='C')

        plt.text(2.0,0., r"$0^\circ$", ha="left", va="center")
        plt.text(1.9,0.45, r"$30^\circ$", ha="left", va="center")
        plt.text(1.4,0.8, r"$60^\circ$", ha="left", va="center")
        plt.text(1.9,-0.45, r"$-30^\circ$", ha="left", va="center")
        plt.text(1.4,-0.8, r"$-60^\circ$", ha="left", va="center")
        plt.text(2.0, -0.15, r"$0^\circ$", ha="center", va="center")
        plt.text(1.333, -0.15, r"$60^\circ$", ha="center", va="center")
        plt.text(.666, -0.15, r"$120^\circ$", ha="center", va="center")
        plt.text(0.0, -0.15, r"$180^\circ$", ha="center", va="center") 
        plt.text(-.666, -0.15, r"$240^\circ$", ha="center", va="center")
        plt.text(-1.333, -0.15, r"$300^\circ$", ha="center", va="center")
        plt.text(-2.0, -0.15, r"$360^\circ$", ha="center", va="center")

        probs = hp.pixelfunc.ud_grade(self.skymap, 64)
        probs = probs/np.sum(probs)
        pixels = np.arange(probs.size)
        sample_points = np.array(hp.pix2ang(self.nside,pixels)).T
        min_prob = np.amin(probs)

        sorted_samples = list(reversed(list(sorted(probs))))
        ### plot 90% containment contour of LIGO PDF
        levels = [0.9]
        theta, phi = self.plot_contours(levels,probs)
        hp.projplot(theta[0],phi[0],linewidth=1,c='k')
        for i in range(1,len(theta)):
            hp.projplot(theta[i],phi[i],linewidth=1,c='k')

        plt.title('GW Skymap',fontsize=14)
        plt.legend(loc=1,bbox_to_anchor=(1.07,1.05))
        plt.savefig(self.dirname + '/unblinded_skymap.png',bbox_inches='tight')

    def plot_skymap_zoom(self,events,ra,dec):
        r'''Make a zoomed in portion of a skymap with
        all neutrino events within a certain range

        Parameters:
        ----------
        ra: float
            RA value (in radians) of center of zoom plot
        dec: float
            Dec value (in radians) of center of zoom plot

        '''

        col_num = 5000
        seq_palette = sns.color_palette("Spectral", col_num)
        lscmap = mpl.colors.ListedColormap(seq_palette)

        rel_t = np.array((events['time'] - self.start) * col_num / (self.stop - self.start), dtype = int)
        cols = [seq_palette[j] for j in rel_t]

        self.plot_zoom(self.skymap, ra, dec, "Zoomed Scan Results", range = [0,10], reso=3.)

        if (self.stop - self.start) <= 50.:
            self.plot_events(events['dec'], events['ra'], events['sigma']*self._angScale, ra, dec, 2*6, sigma_scale=1.0,
                    constant_sigma=False, same_marker=True, energy_size=True, col = cols)
        else:
            #CHANGE THIS TO ONLY PLOT EVENTS, NOT CONTOURS
            self.plot_events(events['dec'], events['ra'], events['sigma']*self._angScale, ra, dec, 2*6, sigma_scale=None,
                    constant_sigma=False, same_marker=True, energy_size=True, col = cols)
                    #sigma_scale = 4.5 is set to match the size in degrees for the given plot size


        plt.scatter(0,0, marker='*', c = 'k', s = 130, label = 'Max TS of Scan')

        probs = hp.pixelfunc.ud_grade(self.skymap, 64,power=-2)
        pixels = np.arange(probs.size)
        sample_points = np.array(hp.pix2ang(self.nside,pixels)).T
        min_prob = np.amin(probs)

        sorted_samples = list(reversed(list(sorted(probs))))
        ## plot 90% containment contour of LIGO PDF
        levels = [0.9]
        theta, phi = self.plot_contours(levels,probs)
        handle = hp.projplot(theta[0],phi[0],linewidth=1,c='k')
        for i in range(1,len(theta)):
            hp.projplot(theta[i],phi[i],linewidth=1,c='k')

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        i3_event_pt = mlines.Line2D([], [], color = 'r', marker = 'x', markersize = 6, linewidth = 0)
        i3_event_contour = mlines.Line2D([], [], marker = 'o', markeredgecolor = 'r', 
                                         linewidth = 0, markersize = 25, markerfacecolor = 'w')
        patch = (i3_event_contour, i3_event_pt)
        handles.append(patch) 
        labels.append('IC Event (90\% C.L.)')
        handles.append(handle)
        labels.append('GW 90\% C.L.')
        plt.legend(handles=handles, labels = labels, loc = 2, ncol=2, mode = 'expand', fontsize = 15.5, framealpha = 0.95)
        self.plot_color_bar(range=[0,6], cmap=lscmap, col_label=r"IceCube Event Time",
                    offset=-45, labels = [r'-500 s', r'+500 s'])
        plt.savefig(self.dirname + '/unblinded_skymap_zoom.png',bbox_inches='tight')

    def plot_zoom(self,scan, ra, dec, title, reso=3, var="pVal", range=[0, 6],cmap=None):
        if cmap is None:
            pdf_palette = sns.color_palette("Blues", 500)
            cmap = mpl.colors.ListedColormap(pdf_palette)
        hp.gnomview(scan, rot=(np.degrees(ra), np.degrees(dec), 0),
                        cmap=cmap,
                        max=max(scan)*0.1,
                        reso=reso,
                        title=title,
                        notext=True,
                        cbar=False
                        #unit=r""
                        )

        plt.plot(4.95/3.*reso*np.radians([-1, 1, 1, -1, -1]), 4.95/3.*reso*np.radians([1, 1, -1, -1, 1]), color="k", ls="-", lw=3)
        hp.graticule(verbose=False)
        self.plot_labels(dec, ra, reso)

    def plot_labels(self,src_dec, src_ra, reso):
        """Add labels to healpy zoom"""
        fontsize = 20
        plt.text(-1*np.radians(1.75*reso),np.radians(0), r"%.2f$^{\circ}$"%(np.degrees(src_dec)),
                 horizontalalignment='right',
                 verticalalignment='center', fontsize=fontsize)
        plt.text(-1*np.radians(1.75*reso),np.radians(reso), r"%.2f$^{\circ}$"%(reso+np.degrees(src_dec)),
                 horizontalalignment='right',
                 verticalalignment='center', fontsize=fontsize)
        plt.text(-1*np.radians(1.75*reso),np.radians(-reso), r"%.2f$^{\circ}$"%(-reso+np.degrees(src_dec)),
                 horizontalalignment='right',
                 verticalalignment='center', fontsize=fontsize)
        plt.text(np.radians(0),np.radians(-1.75*reso), r"%.2f$^{\circ}$"%(np.degrees(src_ra)),
                 horizontalalignment='center',
                 verticalalignment='top', fontsize=fontsize)
        plt.text(np.radians(reso),np.radians(-1.75*reso), r"%.2f$^{\circ}$"%(-reso+np.degrees(src_ra)),
                 horizontalalignment='center',
                 verticalalignment='top', fontsize=fontsize)
        plt.text(np.radians(-reso),np.radians(-1.75*reso), r"%.2f$^{\circ}$"%(reso+np.degrees(src_ra)),
                 horizontalalignment='center',
                 verticalalignment='top', fontsize=fontsize)
        plt.text(-1*np.radians(2.3*reso), np.radians(0), r"declination",
                    ha='center', va='center', rotation=90, fontsize=fontsize)
        plt.text(np.radians(0), np.radians(-2*reso), r"right ascension",
                    ha='center', va='center', fontsize=fontsize)

    def plot_color_bar(self,labels=[0.,2.,4.,6.],col_label=r"IceCube Event Time",
                       range=[0,6],cmap=None,offset=-35):
        fig = plt.gcf()
        #ax = fig.add_axes([0.25, -0.03, 0.5, 0.03])
        ax = fig.add_axes([0.95, 0.2, 0.03, 0.6])
        labels = labels
        cb = mpl.colorbar.ColorbarBase(ax, cmap=ps_map if cmap is None else cmap,
                            #norm=mpl.colors.Normalize(vmin=range[0], vmax=range[1]),
                            orientation="vertical")
        #cb.ax.minorticks_on()

        cb.set_label(col_label, labelpad=offset)
        cb.set_ticks([0., 1.])
        cb.set_ticklabels(labels)
        cb.update_ticks()
        #cb.ax.get_xaxis().set_ticklabels(labels)

    def plot_events(self,dec, ra, sigmas, src_ra, src_dec, reso, sigma_scale=5., col = 'k', constant_sigma=False,
                        same_marker=False, energy_size=False, with_mark=True, with_dash=False):
        """Adds events to a healpy zoom, get events from llh."""
        cos_ev = np.cos(dec)
        tmp = np.cos(src_ra - ra) * np.cos(src_dec) * cos_ev + np.sin(src_dec) * np.sin(dec)
        dist = np.arccos(tmp)

        if sigma_scale is not None:
            sigma = np.degrees(sigmas)/sigma_scale
            sizes = 5200*sigma**2
            if constant_sigma:
                sizes = 20*np.ones_like(sizes)
            if with_dash:
                hp.projscatter(np.pi/2-dec, ra, marker='o', linewidth=2, edgecolor=col, linestyle=':', facecolor="None", s=sizes, alpha=1.0)
            else:
                hp.projscatter(np.pi/2-dec, ra, marker='o', linewidth=2, edgecolor=col, facecolor="None", s=sizes, alpha=1.0)
        if with_mark:
            hp.projscatter(np.pi/2-dec, ra, marker='x', linewidth=2, edgecolor=col, facecolor=col, s=60, alpha=1.0)

    def plot_contours(self,proportions,samples):
        r''' Plot containment contour around desired level.
        E.g 90% containment of a PDF on a healpix map

        Parameters:
        -----------
        proportions: list
            list of containment level to make contours for.
            E.g [0.68,0.9]
        samples: array
            array of values read in from healpix map
            E.g samples = hp.read_map(file)
        Returns:
        --------
        theta_list: list
            List of arrays containing theta values for desired contours
        phi_list: list
            List of arrays containing phi values for desired contours
        '''

        levels = []
        sorted_samples = list(reversed(list(sorted(samples))))
        nside = hp.pixelfunc.get_nside(samples)
        sample_points = np.array(hp.pix2ang(nside,np.arange(len(samples)))).T
        for proportion in proportions:
            level_index = (np.cumsum(sorted_samples) > proportion).tolist().index(True)
            level = (sorted_samples[level_index] + 
                    (sorted_samples[level_index+1] if level_index+1<len(samples) else 0))/2.0
            levels.append(level)
        contours_by_level = meander.spherical_contours(sample_points, samples, levels)
        
        theta_list = []; phi_list=[]
        for contours in contours_by_level:
            for contour in contours:
                theta, phi = contour.T
                phi[phi<0] += 2.0*np.pi
                theta_list.append(theta)
                phi_list.append(phi)

        return theta_list, phi_list

    def make_decPDF(self,name):
        r''' Plot PDF of source declination overlaid with IceCube's
        point source sensitivity

        Parameters:
        name: str
            Name of source
        '''

        sinDec_bins = np.array([-1.,-0.982,-0.965,-0.947,-0.93,-0.867,-0.804, -0.741, -0.678,
                   -0.615,-0.552,-0.489,-0.426,-0.363,-0.3,-0.26111111,-0.22222222,
                   -0.18333333,-0.14444444,-0.10555556,-0.06666667,-0.02777778, 
                   0.01111111,0.05,0.10277778,0.15555556,0.20833333,0.26111111,0.31388889,
                   0.36666667,0.41944444,0.47222222,0.525,0.57777778,0.63055556,0.68333333,
                   0.73611111,0.78888889,0.84166667,0.89444444,0.94722222,1.])

        dec_range = np.linspace(-1,1,35)
        sens = [1.15,1.06,.997,.917,.867,.802,.745,.662,.629,.573,.481,.403,
                .332,.250,.183,.101,.035,.0286,.0311,.0341,.0361,.0394,.0418,
                .0439,.0459,.0499,.0520,.0553,.0567,.0632,.0679,.0732,.0788,.083,.0866]
        sens = np.array(sens)
        sens/=sens.sum()

        bin_centers = (sinDec_bins[:-1] + sinDec_bins[1:]) / 2

        pixels = np.arange(len(self.skymap))
        theta, ra = hp.pix2ang(self.nside,pixels)
        dec = np.pi/2 - theta
        sindec = np.sin(dec)

        pdf = []
        for i in range(sinDec_bins.size-1):
            dec_min = sinDec_bins[i] ; dec_max = sinDec_bins[i+1]
            mask = (sindec<dec_max) & (sindec>dec_min)
            pdf.append(self.skymap[pixels[mask]].sum())

        pdf = np.array(pdf)

        fig,ax1 = plt.subplots()
        ax1.set_xlabel('sin($\delta$)')
        ax1.set_ylabel('Probability')
        ax1.set_xlim(-1,1)
        ax1.plot(bin_centers, pdf, color='C0',label='PDF')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('E$^2$F (GeVcm$^2$)')  # we already handled the x-label with ax1
        ax2.plot(dec_range, sens, color='C1',label='PS Sensitivity')
        ax2.set_yscale('log')
        ax2.set_xlim(-1,1)
        ax2.tick_params(axis='y')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title('%s' % name)
        plt.legend()
        plt.savefig(self.dirname + '/%s_decPDF.png' % name,bbox_to_inches='tight',dpi=200)
        plt.close()

    def contour(self,ra, dec, sigma,nside):
        r''' Function for plotting neutrino contours on skymaps

        Parameters:
        -----------
        ra: ndarray
            Array of ra for events 
        dec: ndarray
            Array of dec for events
        sigma: ndarray
            Array of sigma to make contours around events
        nside:
            nside of healpy map
        Returns:
        --------
        Theta: array
            array of theta values of contour
        Phi: array
            array of phi values of contour 
        '''
        dec = np.pi/2 - dec
        sigma = np.rad2deg(sigma)
        delta, step, bins = 0, 0, 0
        delta= sigma/180.0*np.pi
        step = 1./np.sin(delta)/20.
        bins = int(360./step)
        Theta = np.zeros(bins+1, dtype=np.double)
        Phi = np.zeros(bins+1, dtype=np.double)
        # define the contour
        for j in range(0,bins) :
                phi = j*step/180.*np.pi
                vx = np.cos(phi)*np.sin(ra)*np.sin(delta) + np.cos(ra)*(np.cos(delta)*np.sin(dec) + np.cos(dec)*np.sin(delta)*np.sin(phi))
                vy = np.cos(delta)*np.sin(dec)*np.sin(ra) + np.sin(delta)*(-np.cos(ra)*np.cos(phi) + np.cos(dec)*np.sin(ra)*np.sin(phi))
                vz = np.cos(dec)*np.cos(delta) - np.sin(dec)*np.sin(delta)*np.sin(phi)
                idx = hp.vec2pix(nside, vx, vy, vz)
                DEC, RA = hp.pix2ang(nside, idx)
                Theta[j] = DEC
                Phi[j] = RA
        Theta[bins] = Theta[0]
        Phi[bins] = Phi[0]

        return Theta, Phi

    def pval2sig(self,pvalue):
        r''' Convert pvalue to a significance in sigma

        Parameters:
        -----------
        pvalue: float
            pvalue as a decimal. eg. 0.001

        Returns:
        --------
        sig: float
            Significance in units of sigma; eg. 2.5sigma
        '''
        sig = np.sqrt(2)*erfinv(1-2*pvalue)

        return sig

    def ipixs_in_percentage(self,skymap,percentage):
        """Finding ipix indices confined in a given percentage.
        
        Input parameters
        ----------------
        skymap: ndarray
            array of probabilities for a skymap
        percentage : float
            fractional percentage from 0 to 1  
        Return
        ------- 
        ipix : numpy array
            indices of pixels within percentage containment
        """
    
        sort = sorted(skymap, reverse = True)
        cumsum = np.cumsum(sort)
        index, value = min(enumerate(cumsum),key=lambda x:abs(x[1]-percentage))

        index_hpx = range(0,len(skymap))
        hpx_index = np.c_[skymap, index_hpx]

        sort_2array = sorted(hpx_index,key=lambda x: x[0],reverse=True)
        value_contour = sort_2array[0:index]

        j = 1 
        table_ipix_contour = [ ]

        for i in range (0, len(value_contour)):
            ipix_contour = int(value_contour[i][j])
            table_ipix_contour.append(ipix_contour)
    
        ipix = table_ipix_contour
          
        return np.asarray(ipix,dtype=int)

    def make_gcn(self,name,start,stop,**kwargs):
        r''' Generate GCN based on information from GW follow up
            analysis. 

        Parameters
        ----------
        name: str
            Name of GW event in the form YYMMDD
        start: str
            String of beginning of search window in UTC
        stop: str
            String of end of search window in UTC
        sens: str
            90% sensitivity upper limit
        
        Returns:
        --------
        gcn: str
            Text for GCN
        '''
        events = kwargs.pop('events',None)
        pvalue = kwargs.pop('pvalue',None)
        namelist = name.split('-')
        gw_name = namelist[0]
        noticeID = namelist[1]+'-'+namelist[2]
        if pvalue>0.01:
            with open(self.analysis_path + '/gcn_template_low.txt','r') as gcn_template:

                gcn = gcn_template.read()
                low_sens,high_sens = self.ps_sens_range()
                gcn = gcn.replace('<lowSens>','{:1.3f}'.format(low_sens))
                gcn = gcn.replace('<highSens>','{:1.3f}'.format(high_sens))
                gcn = gcn.replace('<name>',gw_name)
                gcn = gcn.replace('<tstart>',start)
                gcn = gcn.replace('<tstop>',stop)
                gcn = gcn.replace('<noticeID>',noticeID)

            gcn_file = open(self.dirname+'/gcn_%s.txt' % name,'w')
            gcn_file.write(gcn)
            gcn_file.close()

        else:
            significance = '{:1.2f}'.format(self.pval2sig(pvalue))

            info = ' <dt>   <ra>       <dec>          <angErr>                    <p_gwava>                 <p_llama>\n'
            table = ''
            for event in events:
                if event['pvalue']<=0.1:
                    ev_info = info
                    ra = '{:.2f}'.format(np.rad2deg(event['ra']))
                    dec = '{:.2f}'.format(np.rad2deg(event['dec']))
                    sigma = '{:.2f}'.format(np.rad2deg(event['sigma']*2.145966))
                    dt = '{:.2f}'.format((event['time']-self.trigger)*86400.)
                    ev_info = ev_info.replace('<dt>',dt)
                    ev_info = ev_info.replace('<ra>',ra)
                    ev_info = ev_info.replace('<dec>',dec)
                    ev_info = ev_info.replace('<angErr>',sigma)
                    if event['pvalue']<0.0013499:
                        pval_str = '<0.00135'
                        ev_info = ev_info.replace('<p_gwava>',pval_str)
                    else:
                        pval_str = '{:1.3f}'.format(event['pvalue'])
                        ev_info = ev_info.replace('<p_gwava>',pval_str)
                    # ev_info = ev_info.replace('<p_gwava>','{:.3f}'.format(pvalue))
                    table+=ev_info


            num = events['pvalue'][events['pvalue']<=0.1].size
            gcn_file = open(self.dirname+'/gcn_%s.txt' % name,'w')
            with open(self.analysis_path + '/gcn_template_high.txt','r') as gcn_template:

                for line in gcn_template.readlines():
                    line = line.replace('<N>',str(num))
                    line = line.replace('<name>',gw_name)
                    line = line.replace('<noticeID>',noticeID)
                    line = line.replace('<tstart>',start)
                    line = line.replace('<tstop>',stop)
                    if pvalue<0.0013499:
                        pval_str = '<0.00135'
                        line = line.replace('<p_gwava>',pval_str)
                        line = line.replace('<sig_gwava>','>3')
                    else:
                        pval_str = '{:1.3f}'.format(pvalue)
                        line = line.replace('<p_gwava>',pval_str)
                        line = line.replace('<sig_gwava>',significance)

                    if '<dt>' in line:
                        line = table

                    gcn_file.write(line)
                gcn_file.close()

    def write_results(self,dirname,ts,ns,gamma,pvalue,latency=None):
        r''' Write results from analysis in a text file.

        Parameters:
        -----------
        dirname: str
            Name of directory where file will be written
        ts: float
            TS of analysis
        ns: float
            Best fit number of signal events from analysis
        gamma: float
            Best fit spectral index from analysis
        pvalue: float
            Overall pvalue of search
        latency: list of floats, optional
            [time taken for analysis_reports,total time]. 

        '''

        file = open(dirname + '/results.txt','w+')
        file.write('RESULTS\n')
        file.write('---------------------\n\n')
        file.write('TS: %.4f\n' % ts)
        file.write('ns: %.4f\n' % ns)
        file.write('gamma: %.4f\n' % gamma)
        file.write('pvalue: %.4f\n' % pvalue)
        if latency is not None:
            file.write('Time taken for analysis and reports: %.4f\n' % latency[0])
            file.write('Total time taken: %.4f' % latency[1])

        file.close()

