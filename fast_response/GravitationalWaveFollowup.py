from fast_response.FastResponseAnalysis import FastResponseAnalysis

class GW_Followup(FastResponseAnalysis):
    r'''FastResponseAnalysis based class for the Gravitational
    Wave Followup'''
    def __init__(self, location, trigger):
        r'''Constructor

        Args:
            location (str): URL for gravitational wave skymap or path to 
                local fits file
            trigger (str): Trigger time in iso format, eg '2019-03-10 12:00:00' 
        '''
        self.name = kwargs.pop("Name", "FastResponseAnalysis")
        self.alert_event = False
        self.skymap_url = location
        self.smear = False
        skymap_fits, skymap_header = hp.read_map(location)
        self.skymap = skymap_fits
        if hp.pixelfunc.get_nside(self.skymap)!=512:
            self.skymap = hp.pixelfunc.ud_grade(self.skymap,512,power=-2)
        self.nside = hp.pixelfunc.get_nside(self.skymap)
        self.ipix_90 = self.ipixs_in_percentage(self.skymap,0.9)
        self.ra, self.dec, self.extension = None, None, None
        self.source_type = 

    def initialize_llh(self):
        pass