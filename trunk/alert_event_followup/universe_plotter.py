import numpy as np
from glob import glob
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import healpy as hp
import scipy.stats as st
import scipy as sp
import pickle
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
mpl.style.use('/home/apizzuto/Nova/scripts/novae_plots.mplstyle')

skymap_files = glob('/data/ana/realtime/alert_catalog_v2/2yr_prelim/fits_files/Run*.fits.gz')
energy_density = {'transient': {'HB2006SFR': 4.8038e51, 
                        'MD2014SFR': 6.196e51,
                        'NoEvolution': 1.8364e52},
                'steady': {'HB2006SFR': 7.6735e43, 
                        'MD2014SFR': 9.897e+43,
                        'NoEvolution': 2.93335e44}}

class UniversePlotter():
    r'''
    Tool to make some helpful plots
    '''
    def __init__(self, delta_t, data_years, lumi, evol, **kwargs):
        self.delta_t = delta_t
        self.transient = True if self.delta_t is not None else False
        self.data_years = data_years
        self.lumi = lumi
        self.evol = evol
        self.background_median_ts = None
        self.background_median_p = None
        self.smeared = kwargs.pop('smeared', True)
        self.ts_fills = [self.background_median_ts, 50.] if self.transient else [self.background_median_ts, 10.]
        self.lumi_str = {'SC': 'Standard Candle', 'LG': 'LogNormal'}[self.lumi]
        self.evol_str = {'HB2006SFR': 'Hopkins and Beacom 2006 SFR',
                            'MD2014SFR': 'Madau and Dickinson 2014 CSFH'}[self.evol]
        self.ts_path = '/data/user/apizzuto/fast_response_skylab/alert_event_followup/ts_distributions/'
        self.steady_str = '_delta_t_{:.2e}'.format(self.delta_t) if self.transient else '_steady'
        self.time_window_per_year = (365. * 86400.) / (self.delta_t) if self.transient else 1.
        key = 'transient' if self.transient else 'steady'
        self.energy_density = energy_density[key][evol]
        self.no_evol_energy_density = energy_density[key]['NoEvolution']
        self.evol_lumi_str = 'evol_{}_lumi_{}'.format(self.evol, self.lumi)
        self.densities = np.logspace(-11., -6., 21)
        self.luminosities = np.logspace(49, 60, 45) if self.transient else np.logspace(49., 56., 29)
        if self.transient:
            low_energy_msk = self.luminosities * self.delta_t * self.densities[0] < self.energy_density * 10.
            high_energy_msk = self.luminosities * self.delta_t * self.densities[-1] > self.energy_density * 1e-4
            self.luminosities = self.luminosities[low_energy_msk * high_energy_msk]
        self.seconds_per_year = 365.*86400.
        self.med_TS = None
        self.get_labels()

    def two_dim_sensitivity_plot_ts(self, compare=False, log_ts=False):
        r'''Two dimensional contour plot
        TODO: Fix luminosity vs energy,
            - Check the plot_lumis units
            - Include comparison lines
            - Fix bounds
        '''
        if self.background_median_ts is None:
            self.get_overall_background_ts()
        if self.med_TS is None:
            self.get_med_TS()
        fig, ax = plt.subplots(figsize=(8,5), dpi=200)
        fig.set_facecolor('w')
        X, Y = np.meshgrid(np.log10(self.densities), np.log10(self.plot_lumis))
        plot_vals = self.med_TS if not log_ts else np.log10(self.med_TS) 

        cs = ax.contour(X, Y, plot_vals, cmap=self.cmap, #levels=np.linspace(-0.5, 1.3, 11), 
                        #vmin=-0.5, 
                        extend='max')
        csf = ax.contourf(X, Y, plot_vals, cmap=self.cmap, #vmin=-0.5, #levels=np.linspace(-0.5, 1.3, 11), 
                        extend='max')
        cbar = plt.colorbar(csf) 
        cbar_lab = r'Median Stacked TS' if not log_ts else r'$\log_{10}($Median Stacked TS$)$'
        cbar.set_label(cbar_lab, fontsize = 18)
        cbar.ax.tick_params(axis='y', direction='out')
        cs_ts = ax.contour(X, Y, self.lower_10 - self.background_median_ts, colors=['k'], 
                        levels=[0.0], linewidths=2.)
        xs = np.logspace(-11., -6., 1000)
        ys_max = self.no_evol_energy_density / xs / self.seconds_per_year if self.transient else self.no_evol_energy_density / xs
        ys_min = self.energy_density / xs / self.seconds_per_year if self.transient else self.energy_density / xs
        plt.fill_between(np.log10(xs), np.log10(ys_min), np.log10(ys_max), 
                color = 'm', alpha = 0.3, lw=0.0, zorder=10)
        if compare:
            self.compare_other_analyses()
        plt.text(-9, 54.1, 'Diffuse', color = 'm', rotation=-28, fontsize=18)
        plt.text(-10, 51.7, 'Sensitivity', color = 'k', rotation=-28, fontsize=18)
        plt.grid(lw=0.0)
        #plt.ylim(50, 55.5)
        plt.ylabel(self.lumi_label, fontsize = 22)
        plt.xlabel(self.density_label, fontsize = 22)
        plt.show()

    def get_labels(self):
        self.lumi_label = r'$\log_{10}\Big( \frac{\mathcal{E}}{\mathrm{erg}} \Big)$' if self.transient \
                else r'$\log_{10}\Big( \frac{\mathcal{L}}{\mathrm{erg}\;\mathrm{yr}^{-1}} \Big)$'
        self.density_label = r'$\log_{10}\Big( \frac{\dot{\rho}}{ \mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}} \Big)$' if self.transient \
                else r'$\log_{10}\Big( \frac{\rho}{ \mathrm{Mpc}^{-3}} \Big)$'
        self.cmap = ListedColormap(sns.light_palette((210, 90, 60), input="husl"))
        self.plot_lumis = self.luminosities / self.time_window_per_year
        self.scaled_lumi_label = r'$\log_{10}\Big( \frac{\mathcal{E}\dot{\rho} }{\mathrm{erg}\mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}} \Big)$' if self.transient \
                else r'$\log_{10}\Big( \frac{\mathcal{L} \rho}{\mathrm{Mpc}^{-3} \mathrm{erg}\;\mathrm{yr}^{-1}} \Big)$'
        self.dens_with_units = r'$\rho$ (Mpc$^{-3}$ yr$^{-1}$)' if self.transient else r'$\rho$ (Mpc$^{-3}$)'
        self.dens_units = r'Mpc$^{-3}$ yr$^{-1}$' if self.transient else r'Mpc$^{-3}$'

    def get_med_TS(self):
        fmt_path = 'ts_dists_{}year_density_{:.2e}_' + self.evol_lumi_str + \
                        '_manual_lumi_{:.1e}' + self.steady_str + '.npy'
        shape = (self.luminosities.size, self.densities.size)             
        med_TS = np.zeros(shape); lower_10 = np.zeros(shape)
        for ii, lumi in enumerate(self.luminosities):
            for jj, dens in enumerate(self.densities):
                test_en = lumi*dens*self.delta_t if self.transient else lumi*dens
                if test_en > self.energy_density*5.:
                    lower_10[ii, jj] = self.ts_fills[1]
                    med_TS[ii, jj] = self.ts_fills[1]
                elif test_en < self.energy_density*1e-4:
                    lower_10[ii, jj] = self.background_lower_10_ts
                    med_TS[ii, jj] = self.background_median_ts
                else:
                    try:
                        trials = np.load(self.ts_path \
                            + fmt_path.format(self.data_years, dens, lumi))
                        lower_10[ii, jj] = np.percentile(trials[0], 10.)
                        med_TS[ii, jj] = np.median(trials[0])
                    except IOError, e:
                        lower_10[ii, jj] = np.nan
                        med_TS[ii, jj] = np.nan
        med_TS = np.where(np.isnan(med_TS), self.background_median_ts, med_TS)
        lower_10 = np.where(np.isnan(lower_10), self.background_lower_10_ts, lower_10)
        self.med_TS = med_TS
        self.lower_10 = lower_10

    def rotated_sensitivity_plot_ts(self, log_ts=False):
        if self.background_median_ts is None:
            self.get_overall_background_ts()
        if self.med_TS is None:
            self.get_med_TS()
        #HELP FIX OOP NOPE
        fig, ax = plt.subplots(figsize=(8,5), dpi=200)
        fig.set_facecolor('w')
        X, Y = np.meshgrid(self.densities, self.plot_lumis)
        Y *= X #Scale by the densities
        X = np.log10(X); Y = np.log10(Y)
        plot_vals = self.med_TS if not log_ts else np.log10(self.med_TS) 

        cs = ax.contour(X, Y, plot_vals, cmap=self.cmap, #levels=np.linspace(-0.5, 1.3, 11), 
                        #vmin=-0.5, 
                        extend='max')
        csf = ax.contourf(X, Y, plot_vals, cmap=self.cmap, #vmin=-0.5, #levels=np.linspace(-0.5, 1.3, 11), 
                        extend='max')
        cbar = plt.colorbar(csf) 
        cbar_lab = r'Median Stacked TS' if not log_ts else r'$\log_{10}($Median Stacked TS$)$'
        cbar.set_label(cbar_lab, fontsize = 18)
        cbar.ax.tick_params(axis='y', direction='out')
        cs_ts = ax.contour(X, Y, self.lower_10 - self.background_median_ts, colors=['k'], 
                        levels=[0.0], linewidths=2.)
        xs = np.logspace(-11., -6., 1000)
        ys_max = self.no_evol_energy_density / xs / self.seconds_per_year if self.transient else self.no_evol_energy_density / xs
        ys_min = self.energy_density / xs / self.seconds_per_year if self.transient else self.energy_density / xs
        plt.fill_between(np.log10(xs), np.log10(ys_min*xs), np.log10(ys_max*xs), 
                color = 'm', alpha = 0.3, lw=0.0, zorder=10)
        #if compare:
        #    self.compare_other_analyses()
        plt.text(-10, np.log10(np.max(ys_max*xs)*1.1), 'Diffuse', color = 'm', rotation=0, fontsize=18)
        plt.text(-10, np.log10(np.min(ys_min*xs)*0.2), 'Sensitivity', color = 'k', rotation=0, fontsize=18)
        plt.grid(lw=0.0)
        plt.ylim(np.log10(np.min(ys_min*xs)*3e-2), np.log10(np.max(ys_max*xs)*2))
        plt.ylabel(self.scaled_lumi_label, fontsize = 22)
        plt.xlabel(self.density_label, fontsize = 22)
        plt.show()

    def two_dim_sensitivity_plot_p(self):
        pass

    def one_dim_ts_distributions(self, only_gold=False, in_ts = True, log_ts=True):
        r'''Assuming that the diffuse flux is saturated,
        show brazil band plot that scans over density
        and shows the TS distributions'''
        ts_or_p = 0 if in_ts else 1
        ts_inds = (0, 2) if not only_gold else (1, 3)
        levels = []; dens = []
        for density in self.densities:
            try:
                ts = np.load(self.ts_path + \
                    'ts_dists_{}year_density_{:.2e}_'.format(self.data_years, density) + \
                    self.evol_lumi_str + self.steady_str + '.npy')
            except Exception, e:
                continue
            dens.append(density)
            levels.append(np.percentile(ts[ts_inds[ts_or_p]], [5, 25, 50, 75, 95]))
        levels = np.array(levels).T
        fig = plt.figure(dpi=150, figsize=(8,5))
        fig.set_facecolor('w')
        plt.fill_between(dens, levels[0], levels[-1], alpha = 0.5, 
                            color = sns.xkcd_rgb['light navy blue'], 
                            linewidth = 0.0, label = 'Central 90\%')
        plt.fill_between(dens, levels[1], levels[-2], alpha = 0.75, 
                            color = sns.xkcd_rgb['light navy blue'], 
                            linewidth = 0.0, label = 'Central 50\%')
        plt.plot(dens, levels[2], color = sns.xkcd_rgb['light navy blue'])
        plt.title("{}, {}".format(self.lumi_str, self.evol_str))
        plt.xlabel(self.dens_with_units)
        ylab = 'TS' if in_ts else 'Binomial p'
        plt.ylabel('TS')
        plt.xscale('log')
        loc = 1 if in_ts else 4
        plt.legend(loc=loc, frameon=False)
        if log_ts:
            plt.yscale('log')
        plt.show()

    def ts_and_ps_plot(self, only_gold=False, log_ts=True):
        r'''Make TS distributions for density, luminosity 
        pairs that saturate the diffuse flux'''
        ts_inds = (0, 2) if not only_gold else (1, 3)
        fig, axs = plt.subplots(ncols=2, nrows=1, dpi=200, sharey=True, figsize=(10,4))
        plt.subplots_adjust(wspace=0.08)
        for density in self.densities[::4]:
            try:
                ts = np.load(self.ts_path + 'ts_dists_{}year_density_{:.2e}_'.format(self.data_years, density)
                                + self.evol_lumi_str + self.steady_str + '.npy')
            except IOError, e:
                continue
            ts_bins = np.logspace(-1., 2., 31) if log_ts else np.linspace(0., 15., 31)
            axs[0].hist(ts[ts_inds[0]], bins = ts_bins, label = r'$\rho = $' 
                    + '{:.1e}'.format(density) + ' ' + self.dens_units, 
                    histtype='step', lw=2.5, weights = [1./len(ts[0])]*len(ts[0]))
            axs[1].hist(ts[ts_inds[1]], bins = np.logspace(-20., 0, 31), label = r'$\rho = $' 
                    + '{:.1e}'.format(density) + ' ' + self.dens_units, 
                    histtype='step', lw=2.5, weights = [1./len(ts[2])]*len(ts[2]))
        fig.suptitle(self.lumi_str + '\n'+ self.evol_str, y=1.03)
        axs[0].set_ylabel('Probability')
        axs[0].set_xlabel('TS')
        axs[1].set_xlabel(r'Binomial $p$')
        if log_ts:
            axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[0].set_yscale('log'); axs[1].set_yscale('log')
        axs[0].set_ylim(8e-3, 7e-1); axs[1].set_ylim(8e-3, 7e-1)
        axs[1].legend(loc=(1.01, 0.1))
        plt.show()

    def get_overall_background_ts(self, n_trials=1000):
        if self.background_median_ts is not None:
            return self.background_median_ts
        sigs = []
        for ind in range(len(skymap_files)):
            if ind == 19:
                sig = 0.
            else:
                skymap_fits, skymap_header = hp.read_map(skymap_files[ind], h=True, verbose=False)
                skymap_header = {name: val for name, val in skymap_header}
                sig = skymap_header['SIGNAL']
            sigs.append(sig)
        self.sigs = np.array(sigs)
        bg_trials = '/data/user/apizzuto/fast_response_skylab/alert_event_followup/analysis_trials/bg/'
        TSs = []
        for ind in range(len(skymap_files)):
            if ind == 19:
                ts = [0.]*n_trials
                TSs.append(ts)
            else:
                smeared_str = 'smeared/' if self.smeared else 'norm_prob/'
                if self.transient:
                    trials_file = glob(bg_trials + smeared_str 
                                + 'index_{}_*_time_{:.1f}.pkl'.format(ind, self.delta_t))[0]
                    trials = np.load(trials_file)
                    ts = np.random.choice(trials['ts_prior'], size=n_trials)
                else:
                    trials_files = glob(bg_trials + smeared_str 
                                + 'index_{}_steady_seed_*.pkl'.format(ind))
                    trials = []
                    for f in trials_files:
                        trials.extend(np.load(f)['TS'])
                    ts = np.random.choice(np.array(trials), size=n_trials)
                TSs.append(ts)
        TSs = np.array(TSs)
        stacked_ts = np.multiply(TSs, self.sigs[:, np.newaxis])
        stacked_ts = np.sum(stacked_ts, axis=0) / (stacked_ts.shape[0] - 1.) #-1 because we skip one of the maps
        self.background_median_ts = np.median(stacked_ts)
        self.background_lower_10_ts = np.percentile(stacked_ts, 10.)
        self.stacked_ts = stacked_ts
        return self.background_median_ts

    def inject_and_fit(self, dens, lumi):
        pass

    def upper_limit(self, TS):
        pass

    def compare_other_analyses(self):
        if self.transient:
            for key in ['GRB_lims']:
                tmp = np.log10(zip(*nora_comparison[key]))
                plt.plot(tmp[0], tmp[1], color = 'grey')
            plt.text(-10., 52.55, 'Nora 100 s transients (5 yr.)', 
                            color='grey', rotation=-32, fontsize=18)
        else:
            #COMPARE TO 7 YEAR PS PAPER
            pass

    def fit_coverage_plot(self, dens, lumi):
        pass

nora_comparison = {}
for key in ['GRB_lims', 'GRB_diffuse', 'CCSN_lims', 'CCSN_diffuse']:
    nora_tmp = np.genfromtxt('/data/user/apizzuto/fast_response_skylab/alert_event_followup/effective_areas_alerts/Nora_{}.csv'.format(key),
                            delimiter=', ')
    nora_comparison[key] = nora_tmp