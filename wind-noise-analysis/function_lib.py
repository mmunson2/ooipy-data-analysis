#!/usr/bin/env python
# coding: utf-8

# In[1]:


## functions for precipitation analysis
import numpy as np
import json
from matplotlib import pyplot as plt
from obspy import read,Stream, Trace
from obspy.core import UTCDateTime
import math as M
from matplotlib import mlab
from matplotlib.colors import Normalize
import requests
from lxml import html
from scipy import signal
import matplotlib.colors as colors
import datetime
from scipy import signal
import requests
import urllib
import datetime
import time
import pandas as pd
#!pip install thredds_crawler
#!sudo python3 -m pip install thredds_crawler
from thredds_crawler.crawl import Crawl

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


def remove_adcp_energy(samples, N, plot=False):
    N = 2**11
    i = 0
    threshold = 10**6.5
    delete_list = []
    samples_no_adcp = np.copy(samples)
    
    while i < len(samples):
        if i+N >= len(samples):
            p = sum(samples[len(samples)-N-1:len(samples)-1]**2)/N
            if p >= threshold:
                delete_list.append(range(len(samples)-N-1,len(samples)-1))
                #samples = np.delete(samples, list(range(len(samples)-N-1,len(samples)-1)))
        else:
            p = sum(samples[i:i+N]**2)/N
            if p >= threshold:
                delete_list.append(list(range(i,i+N)))
                #samples = np.delete(samples, list(range(i,i+N)))
        i += N
    
    if plot:
        for i in range(len(delete_list)):
            for k in delete_list[i]:
                samples_no_adcp[k] = 0
    else: samples_no_adcp = np.delete(samples_no_adcp, delete_list)
        
    return samples_no_adcp


# In[18]:


def web_crawler_mooring(beginDT, endDT, location='shelf', method='telemetered'):
    USERNAME = 'OOIAPI-HG9FXL99B5WYRI'
    TOKEN =  'TEMP-TOKEN-NHZIN9M7GI4TY3'
    #Sensor Inventory
    SENSOR_BASE_URL = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'
    # Instrument Information
    if location == 'shelf':
        site = 'CE02SHSM'
    elif location == 'offshore':
        site = 'CE04OSSM'
    node = 'SBD11'
    instrument = '06-METBKA000'
    if method == 'telemetered':
        stream = 'metbk_a_dcl_instrument'
    elif method == 'recovered_host':
        stream = 'metbk_a_dcl_instrument_recovered'

    data_request_url ='/'.join((SENSOR_BASE_URL,site,node,instrument,method,stream))

    params = {
      'beginDT':beginDT,
      'endDT':endDT,
      'format':'application/csv',
      'include_provenance':'false',
      'include_annotations':'false',  
    }
    r = requests.get(data_request_url, params=params, auth=(USERNAME, TOKEN))
    
    dataraw = r.json()
    print(method)
    print(dataraw)
    
    #This is the part that checls to ensure the link is ready to go.
    check_complete = dataraw['allURLs'][1] + '/status.txt'
    for i in range(10000): 
        r = requests.get(check_complete)
        if r.status_code == requests.codes.ok:
            print('request completed')
            break
        else:
            time.sleep(.5)
            
    #This part then finds and downloads the requested csv file.
    url = dataraw['allURLs'][0]
    c = Crawl(url, select=['.*\.csv$'], debug=False)
    urls = [s.get("url") for d in c.datasets for s in d.services if s.get("service").lower()== "httpserver"]
    urlsrev = [url for url in reversed(urls)]
    return urlsrev


# In[9]:


# web crawler noise data  
def web_crawler_noise(day_str, node):
    if node == '/LJ01D': #LJ01D'  Oregon Shelf Base Seafloor
        array = '/CE02SHBP'
        instrument = '/11-HYDBBA106'
    if node == '/LJ01A': #LJ01A Oregon Slope Base Seafloore
        array = '/RS01SLBS'
        instrument = '/09-HYDBBA102'
    if node == '/PC01A': #Oregan Slope Base Shallow
        array = '/RS01SBPS'
        instrument = '/08-HYDBBA103'
    if node == '/PC03A': #Axial Base Shallow Profiler
        array = '/RS03AXPS'
        instrument = '/08-HYDBBA303'
    if node == '/LJ01C':
        array = '/CE04OSBP'
        instrument = '/11-HYDBBA105'
        

    mainurl = 'https://rawdata.oceanobservatories.org/files'+array+node+instrument+day_str
    mainurlpage =requests.get(mainurl)
    webpage = html.fromstring(mainurlpage.content)
    suburl = webpage.xpath('//a/@href')

    FileNum = len(suburl) 
    timestep = 5 #save results every 5 seceonds (no overlap)

    data_url_list = []
    for filename in suburl[6:FileNum]:
        data_url_list.append(str(mainurl + filename[2:]))
        
    return data_url_list


# In[11]:


def get_noise_data(start_time, Nh, Nmin, Ns, node='/LJ01D', stop_time = None):
    
    if stop_time == None:
        duration = 3600*Nh + 60*Nmin + Ns
        stop_time = start_time + duration
 
    fmin = 20
    fmax = 30000.0
    
    # get URLs
    day_start = UTCDateTime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
    data_url_list = web_crawler_noise(day_start.strftime("/%Y/%m/%d/"), node)
    day_start = day_start + 24*3600
    while day_start < stop_time:
        data_url_list.extend(web_crawler_noise(day_start.year, day_start.month, day_start.day, node))
     
    st_all = None

    # only acquire data for desired time
    for i in range(len(data_url_list)):
        # get UTC time of current and next item in URL list
        utc_time_url_start = UTCDateTime(data_url_list[i].split('YDH-')[1].split('.mseed')[0])
        if i != len(data_url_list) - 1:
            utc_time_url_stop = UTCDateTime(data_url_list[i+1].split('YDH-')[1].split('.mseed')[0])
        else: 
            utc_time_url_stop = UTCDateTime(data_url_list[i].split('YDH-')[1].split('.mseed')[0])
            utc_time_url_stop.hour = 23
            utc_time_url_stop.minute = 59
            utc_time_url_stop.second = 59
            utc_time_url_stop.microsecond = 999999
            
        # if current segment contains desired data, store data segment
        if (utc_time_url_start >= start_time and utc_time_url_start < stop_time) or         (utc_time_url_stop >= start_time and utc_time_url_stop < stop_time) or         (utc_time_url_start <= start_time and utc_time_url_stop >= stop_time):
            st = read(data_url_list[i],apply_calib=True)
            
            # slice stream to get desired data
            st = st.slice(UTCDateTime(start_time), UTCDateTime(stop_time))
            
            print(st[0].stats.starttime)
            if st_all == None: st_all = st
            else: 
                st_all += st
                st_all.merge(fill_value ='interpolate',method=1)

    try:
        st_all = st_all.split()
        st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
        print(st_all[0].stats)
        return st_all
    except:
        if st_all == None:
            print('No data available for selected time frame.')
        else: print('Other exception')
    


# In[17]:


# get mooring data for given time
def get_mooring_data(url_list):
    data = []
    for url in url_list:
        data.append(pd.read_csv(url))
        
    data_all = pd.concat(data, ignore_index=True)
    return data_all


# In[15]:


# spectrogram
def plot_spectrogram(st,start_sec, end_sec, vmin, vmax, fkhz_min,fkhz_max, adcp=''):
    def _nearest_pow_2(x):
        a = M.pow(2, M.ceil(np.log2(x)))
        b = M.pow(2, M.floor(np.log2(x)))
        if abs(a - x) < abs(b - x):
            return a
        else:
            return b

    spec_data = st.slice(st[0].stats.starttime+start_sec, st[0].stats.starttime+end_sec)
    fs = st[0].stats.sampling_rate
    wlen = 0.056;  # bin size in sec 
    npts = len(spec_data[0])
    end = npts / fs
    nfft = int(_nearest_pow_2(wlen * fs))  # number of fft points of each bin
    print(nfft)
    per_lap = 0.10      # percentage of overlap
    nlap = int(nfft * float(per_lap))   # number of overlapped samples
    
    
    # remove ADCP
    if adcp == 'energy':
        samples = remove_adcp_energy(spec_data[0].data, nfft)
    #elif adcp == 'period':
    #    samples = remove_adcp_periodicity(spec_data[0].data, fs)
    else:
        samples = spec_data[0].data
    
    # using mlab to create the array of spectrogram 
    specgram, freq, time = mlab.specgram(samples,NFFT = nfft,Fs = fs,noverlap = nlap, pad_to = None)
    specgram = 10 * np.log10(specgram[1:, :]) +169-128.9
    specgram = np.flipud(specgram)
    freq = freq[1:] / 1e3  # Convert Frequency to kHz
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0
    freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
    time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
    extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                      freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
    # colormap setting
    #vmin = 0.50  # default should be 0 to start from the min number of the spectrgram
    #vmax = 0.90  # default should be 1 to end at the max number of the spectrgram
    _range = float(specgram.max() - specgram.min())
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax)  # to scale a 2-D float X input to the (0, 1) range for input to the cmap

    # plot spectrogram
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.imshow(specgram, interpolation="nearest", extent=extent, norm=norm, cmap='viridis')
    dpi = fig.get_dpi()
    fig.set_size_inches(512/float(dpi),512/float(dpi))
    ax.axis('tight')
    ax.set_xlim(0, end)
    ax.set_ylim(fkhz_min,fkhz_max)
    ax.grid(False)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_title(spec_data[0].stats.starttime)
    cbar = fig.colorbar(cax)


# In[16]:


# convert seconds to datetime
def ntp_seconds_to_datetime(ntp_seconds):
    ntp_epoch = datetime.datetime(1900, 1, 1)
    unix_epoch = datetime.datetime(1970, 1, 1)
    ntp_delta = (unix_epoch - ntp_epoch).total_seconds()
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)

