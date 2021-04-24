import numpy as np
from datetime import datetime, timedelta
from ooipy.request.hydrophone_request import get_acoustic_data
import ooipy
from ooipy.hydrophone.basic import Spectrogram
from ooipy.hydrophone.basic import Psd
import os
import pickle
import logging
from datetime import datetime
import time
import numpy.ma as ma
from segment_data import SegmentData

from ooipy.tools.ooiplotlib import plot
from ooipy.tools.ooiplotlib import plot_spectrogram
from ooipy.tools.ooiplotlib import plot_psd

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from ooipy.hydrophone.basic import Spectrogram, Psd, HydrophoneData
from matplotlib.colors import Normalize
import matplotlib.dates as mdates


def open_spectrogram(file_name):

    spectrogram_dictionary = pickle.load(open(file_name, "rb"))

    spectrogram = Spectrogram(spectrogram_dictionary['t'],
                              spectrogram_dictionary['f'],
                              spectrogram_dictionary['spectrogram'])

    return spectrogram


IGNORE_FILES = [".DS_Store", "meta.pickle", "meta.txt", "profile.pickle",
                "profile.txt", "segment_data.pickle", "log.txt"]


def get_file_tuple_list():
    dates = []

    for file in os.listdir():

        ignore = False

        for ignored_file in IGNORE_FILES:
            if (file == ignored_file):
                ignore = True
                break

        if ignore:
            continue

        date = datetime.strptime(file, "%Y-%m-%d %H:%M:%S_data.pickle")

        entry = (date, file)

        dates.append(entry)

    dates = sorted(dates)

    return dates


def concatenate_month(file_tuple_list):
    full_spectrogram = None
    count = 0

    for entry in file_tuple_list:
        (date, file) = entry

        spectrogram = open_spectrogram(file)

        if (full_spectrogram == None):
            full_spectrogram = spectrogram
        else:
            full_spectrogram.time = np.append(full_spectrogram.time,
                                              spectrogram.time)
            full_spectrogram.values = np.concatenate(
                (full_spectrogram.values, spectrogram.values), axis=0)

        count += 1

        if (count == round(len(file_tuple_list) * 0.25)):
            print("25% Complete")
        if (count == round(len(file_tuple_list) * 0.5)):
            print("50% Complete")
        if (count == round(len(file_tuple_list) * 0.75)):
            print("75% Complete")

    full_spectrogram.save("data.pickle")
    print("Spectrogram Concatenated and Saved")

    return full_spectrogram


def downsize_spectrogram(spectrogram):
    count = 0

    new_spectrogram = np.ones(
        shape=[len(spectrogram.values), len(spectrogram.values[0])],
        dtype=np.float16)

    for j in range(len(spectrogram.values)):

        current_slice = spectrogram.values[j]

        for i in range(0, len(current_slice)):
            originalVal = current_slice[i]

            smallerVal = np.float16(originalVal)

            new_spectrogram[j][i] = smallerVal

        count += 1

        if (count == round(len(spectrogram.values) * 0.25)):
            print("25% Complete")
        if (count == round(len(spectrogram.values) * 0.5)):
            print("50% Complete")
        if (count == round(len(spectrogram.values) * 0.75)):
            print("75% Complete")

    spectrogram.values = new_spectrogram

    return spectrogram


def has_concatenated_data(month_directory):
    os.chdir(month_directory)

    for file in os.listdir():

        if (file == "data.pickle"):
            os.chdir("..")
            return True

    os.chdir("..")
    return False


def has_downsized_data(month_directory):
    os.chdir(month_directory)

    for file in os.listdir():

        if (file == "reduced_data.pickle"):
            os.chdir("..")
            return True

    os.chdir("..")
    return False


if __name__ == '__main__':

    os.chdir("data")

    for directory in os.listdir():

        if(not os.path.isdir(directory)):
            continue

        if(has_downsized_data(directory)):
            continue

        if (has_concatenated_data(directory)):

            if (has_downsized_data(directory)):
                continue
            else:
                os.chdir(directory)

                full_spectrogram = open_spectrogram("data.pickle")

                print("Compressing " + directory)

                small_spectrogram = downsize_spectrogram(full_spectrogram)

                small_spectrogram.save("reduced_data.pickle")

                os.chdir("..")

                continue

        os.chdir(directory)

        file_tuple_list = get_file_tuple_list()

        print("Concatenating " + directory)

        full_spectrogram = concatenate_month(file_tuple_list)

        print("Compressing " + directory)

        small_spectrogram = downsize_spectrogram(full_spectrogram)

        small_spectrogram.save("reduced_data.pickle")

        os.chdir("..")