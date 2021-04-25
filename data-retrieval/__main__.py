
import os
from datetime import datetime, timedelta
from datetime import datetime
from extended_spectrogram import ExtendedSpectrogram


def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + timedelta(
        days=4)

    return next_month


if __name__ == '__main__':

    start = datetime(-1, -1, -1, 0, 0, 0)
    end = datetime(-1, -1, -1, 0, 0, 0)
    hydrophone = 'INVALID'

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

        # This part only works on the remote machines
        os.chdir(os.path.expanduser("~"))
        os.chdir('code')

        print("Start Time: ", start_times[i])
        print("End Time: ", end_times[i])
        print("Name: ", names[i])
        print("_____")

        spec = ExtendedSpectrogram(start_times[i],
                                   end_times[i],
                                   names[i],
                                   15,
                                   node=hydrophone)


    os.system('sudo systemctl poweroff')


