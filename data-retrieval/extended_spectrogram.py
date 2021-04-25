
import os
import numpy as np
from datetime import datetime, timedelta
import ooipy
from ooipy.request.hydrophone_request import get_acoustic_data
from ooipy.hydrophone.basic import Spectrogram
import pickle
import logging
from segment_data import SegmentData
from meta_file import MetaFile
from profiling_file import ProfilingFile


class ExtendedSpectrogram:
    # _________________________________________________________________________
    # Filename defaults
    DATA_DIRECTORY_NAME = 'data'
    DATA_FILE_NAME = 'data.pickle'
    SEGMENT_DATA_FILE_NAME = 'segment_data.pickle'
    LOG_FILE_NAME = 'log.txt'

    # _________________________________________________________________________
    # Hydrophone_data defaults

    DEFAULT_NODE = '/LJ01C'  # Default is Oregon offshore base seafloor
    DEFAULT_FMIN = None
    DEFAULT_FMAX = None

    # _________________________________________________________________________
    # Compute_Psd defaults

    DEFAULT_WIN = 'hann'
    DEFAULT_L = 4096
    DEFAULT_OVERLAP = 0.5
    DEFAULT_AVG_METHOD = 'median'
    DEFAULT_INTERPOLATE = None
    DEFAULT_SCALE = 'log'

    # Modified 12/20/2020: Added defaults for new Compute_Psd
    DEFAULT_AVG_TIME = 1.0
    DEFAULT_PERCENTILE = 0.8

    def format_kw_args(self):

        if 'node' not in self.kwargs:
            self.kwargs['node'] = self.DEFAULT_NODE

        if 'fmin' not in self.kwargs:
            self.kwargs['fmin'] = self.DEFAULT_FMIN

        if 'fmax' not in self.kwargs:
            self.kwargs['fmax'] = self.DEFAULT_FMAX

        if 'win' not in self.kwargs:
            self.kwargs['win'] = self.DEFAULT_WIN

        if 'L' not in self.kwargs:
            self.kwargs['L'] = self.DEFAULT_L

        if 'overlap' not in self.kwargs:
            self.kwargs['overlap'] = self.DEFAULT_OVERLAP

        if 'avg_method' not in self.kwargs:
            self.kwargs['avg_method'] = self.DEFAULT_AVG_METHOD

        if 'interpolate' not in self.kwargs:
            self.kwargs['interpolate'] = self.DEFAULT_INTERPOLATE

        if 'scale' not in self.kwargs:
            self.kwargs['scale'] = self.DEFAULT_SCALE

        if 'avg_time' not in self.kwargs:
            self.kwargs['avg_time'] = self.DEFAULT_AVG_TIME

        if 'percentile' not in self.kwargs:
            self.kwargs['percentile'] = self.DEFAULT_PERCENTILE

    def get_hydrophone_data(self, start_time, end_time):

        node = self.kwargs.get("node")
        fmin = self.kwargs.get("fmin")
        fmax = self.kwargs.get("fmax")

        segment = ooipy.request.hydrophone_request.get_acoustic_data(
                starttime=start_time, endtime=end_time, node=node,
                fmin=fmin, fmax=fmax, append=False, data_gap_mode=2,
                mseed_file_limit=100)

        return segment

    def compute_psd(self, segment):

        win = self.kwargs.get("win")
        length = self.kwargs.get("L")
        avg_time = self.kwargs.get("avg_time")
        percentile = self.kwargs.get("percentile")
        overlap = self.kwargs.get("overlap")

        # avg_method = self.kwargs.get("avg_method")
        # interpolate = self.kwargs.get("interpolate")
        # scale = self.kwargs.get("scale")

        # Modified 12/20/2020: Switched to compute_spectrogram_wp
        spec2 = segment.compute_spectrogram_wp(
            win=win,
            L=length,
            avg_time=avg_time,
            overlap=overlap,
            verbose=False,
            percentile=percentile,
            )

        # compressing spectrogam by averaging (mean) over each 60s interval
        spec_1m = []
        spec = ooipy.Spectrogram(time=np.linspace(0, 59, 60), freq=spec2.freq,
                                 values=None)

        for i in range(60):
            avg_psd = 10 * np.log10(
                np.mean(10 ** (spec2.values[i * 60:(i + 1) * 60, :] / 10),
                        axis=0))
            spec_1m.append(avg_psd)
        spec.values = np.array(spec_1m)

        return segment, spec

    @staticmethod
    def open_spectrogram(dir_name):

        spectrogram_dictionary = pickle.load(open(dir_name, "rb"))

        spectrogram = Spectrogram(spectrogram_dictionary['t'],
                                  spectrogram_dictionary['f'],
                                  spectrogram_dictionary['spectrogram'])

        return spectrogram

    def write_to_pickle(self, segment, segment_spectrogram, section_midpoint_time, section_start_time):

        if segment is None:

            data = []

            for i in range(0, 15):
                data.append([np.NaN] * 2049)
        else:
            data = segment_spectrogram.values

        timeArray = []

        # Offset the start time so our segment times are at the 30 second mark
        section_start_time += timedelta(seconds=30)

        # Fill the timeArray at one minute increments
        for i in range(0, len(data)):
            timeArray.append(section_start_time + timedelta(minutes=i))

        new_file_name = str(section_start_time) + "_data.pickle"

        if segment is None:
            next_time = np.array(timeArray)
            freq = self.default_freq_array
            values = np.array(data)
        else:
            next_time = np.array(timeArray)
            freq = np.array(segment_spectrogram.freq)
            values = np.array(data)

        spectrogram = Spectrogram(next_time, freq, values)
        spectrogram.save(new_file_name)

        logging.info("Spectrogram file created")


    def write_segment_data(self, segment):

        self.segment_data.open_segment()

        if segment is None:
            self.segment_data.add_entry(segment, 0.0)
        elif type(segment.data) == np.ndarray:
            self.segment_data.add_entry(segment, 1.0)
        elif type(segment.data) == np.ma.core.MaskedArray:

            bool_sum = 0

            for val in segment.data.mask:
                if val:
                    bool_sum += 1

            coverage = bool_sum / len(segment.data.mask)

            self.segment_data.add_entry(segment, coverage)

        self.segment_data.save()

    def create_data_directory(self):

        if not os.path.isdir(self.DATA_DIRECTORY_NAME):
            os.mkdir(self.DATA_DIRECTORY_NAME)

        os.chdir(self.DATA_DIRECTORY_NAME)

    def create_directory(self):

        if os.path.isdir(self.dir_name):
            print(self.dir_name, "already exists. ")
            exit(-1)

        os.mkdir(self.dir_name)

        os.chdir(self.dir_name)

    def initialize_logging(self):
        logging.basicConfig(filename=self.LOG_FILE_NAME,
                            filemode='a', #append to file
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info('Logging started')

    def calculate_segment_count(self):
        start_time_min = self.start_time.timestamp() / 60
        end_time_min = self.end_time.timestamp() / 60

        sections = (end_time_min - start_time_min) / self.segment_length
        sections = round(sections)

        return sections

    def create_spectrogram(self):

        sections = self.calculate_segment_count()

        logging.info("Starting processing loop")

        for sectionCount in range(0, sections):
            print('Downloading Section', sectionCount, 'of', sections)
            logging.info('Downloading Section' + str(sectionCount) + 'of' + str(sections))

            section_start_time = self.start_time + timedelta(minutes=(self.segment_length * sectionCount))
            section_end_time = section_start_time + timedelta(minutes=self.segment_length)
            section_midpoint_time = section_start_time + timedelta(minutes=(self.segment_length / 2))

            # Mark the time that data retrieval starts
            self.profiler.start_data_retrieval()

            segment = self.get_hydrophone_data(section_start_time, section_end_time)
            segment_spectrogram = None

            # Resolving an uncommon OOI error that messes with psd_welch
            if segment is not None and segment.stats.npts < 4096:
                segment = None

            # Append the time it took to process data
            self.profiler.end_data_retrieval()

            if segment is not None:
                self.profiler.start_computation()

                segment, segment_spectrogram = self.compute_psd(segment)

                self.profiler.end_computation()
                logging.info("Data retrieved for segment")
            else:
                logging.info("No data could be retrieved for segment")
                self.missed_segments += 1

            self.write_to_pickle(segment, segment_spectrogram, section_midpoint_time, section_start_time)

            self.write_segment_data(segment)

            # We're not sure if this has an effect, but it's here!
            del segment

        logging.info("Processing loop complete")
        print("Processing Loop Complete")

        self.profiler.save_profiling_data()

        self.meta_file.set_missed_segments(self.missed_segments)
        self.meta_file.save_meta_data()

        logging.shutdown()

    def __init__(self, start_time, end_time, dir_name, segment_length, **kwargs):

        self.start_time = start_time
        self.end_time = end_time
        self.dir_name = dir_name
        self.segment_length = segment_length
        self.kwargs = kwargs
        self.spectrogram = None

        self.missed_segments = 0

        self.default_freq_array = pickle.load(open("default_freq_array.pickle", "rb"))

        # Add default elements to args dictionary
        self.format_kw_args()

        # Switch to the data directory
        self.create_data_directory()
        # Create a new directory for this download
        self.create_directory()
        # Begin Logging
        self.initialize_logging()

        # Initialize segment_data and save a blank file
        self.segment_data = SegmentData(self.SEGMENT_DATA_FILE_NAME)
        self.segment_data.save()

        self.meta_file = MetaFile(self.start_time,
                                  self.end_time,
                                  self.segment_length,
                                  self.calculate_segment_count(),
                                  self.dir_name,
                                  self.kwargs)

        self.meta_file.save_meta_data()

        self.profiler = ProfilingFile(self.dir_name)

        # Begin the download
        self.create_spectrogram()

