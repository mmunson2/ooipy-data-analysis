import sys
import os
import numpy as np
from datetime import datetime, timedelta
import ooipy
from ooipy.request.hydrophone_request import get_acoustic_data
from ooipy.hydrophone.basic import Spectrogram
import pickle
import logging
from datetime import datetime
import time


class SegmentData:

    def __init__(self, filename):
        self.start_times = []
        self.end_times = []
        self.coverageArray = []

        self.filename = filename

    def add_entry(self, segment, coverage):

        if segment is None:
            start_time = np.NAN
            end_time = np.NAN
        else:
            start_time = segment.stats.starttime
            end_time = segment.stats.endtime

        self.start_times.append(start_time)
        self.end_times.append(end_time)
        self.coverageArray.append(coverage)

    def save(self):
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    def open_segment(self):

        segment = pickle.load(open(self.filename, "rb"))

        self.start_times = segment.start_times
        self.end_times = segment.end_times
        self.coverageArray = segment.coverageArray
