
import os
from datetime import datetime, timedelta
from datetime import datetime
from extended_spectrogram import ExtendedSpectrogram


def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + timedelta(
        days=4)

    return next_month


if __name__ == '__main__':

    start = datetime(2019, 1, 1, 0, 0, 0)
    end = datetime(2019, 1, 2, 0, 0, 0)
    hydrophone = 'LJ01C'

    end_of_month = last_day_of_month(start)

    start_times = []
    end_times = []
    names = []

    while end_of_month < end:
        start_times.append(start)
        end_times.append(end_of_month)
        names.append(start.strftime("%B") + "_" + str(start.year))

        start = end_of_month
        end_of_month = last_day_of_month(start)

    start_times.append(start)
    end_times.append(end)
    names.append(start.strftime("%B") + "_" + str(start.year))

    for i in range(0, len(start_times)):

        print("Start Time: ", start_times[i])
        print("End Time: ", end_times[i])
        print("Name: ", names[i])
        print("_____")

        spec = ExtendedSpectrogram(start_times[i],
                                   end_times[i],
                                   names[i],
                                   15,
                                   node=hydrophone)

