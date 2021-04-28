# How to retrieve data:

This module provides functionality for retrieving large amounts of OOI data.

## Background

Server downloads tend to fail for hydrophone data of length greater than 1 hour.
For this reason, downloads are split into 15 minute segments by default. These
files are saved individually to provide maximum protection against corruption in
the event that the hard drive runs out of space. A separate concatenation
python file is used to concatenate the downloaded data.

## Data size and processing length

Note that depending on the length of data requested, the functionality in this
module may take a _very_ long time to execute. The generated .pickle files
may exceed 100 GB in size. It is recommended to try a test download of only
a few hours to profile a system. For large data downloads, it is recommended
to use multiple virtual machines to divide up data processing.

## Configuring \_\_main\_\_.py

\_\_main\_\_.py is the executable to call when beginning the data
download. The start time, end time, and hydrophone node to download
from must first be specified. Below is an example of how to retrieve one
day of data from January 1st 2019 from the Oregon Offshore hydrophone.

    start = datetime(2019, 1, 1, 0, 0, 0)
    end = datetime(2019, 1, 2, 0, 0, 0)
    hydrophone = 'LJ01C'

These parameters are currently
hard-coded into the \_\_main\_\_.py file. They will be converted into
function arguments at a later date. The program will create a data directory
inside the current working directory, then a directory for the month of data
that's being created (data/January_2019 for this example). To prevent accidental
overwrites, the program will refuse to execute if a month directory already
exists.

## Concatenating data 

concatenate_data.py is a separate executable used to concatenate the individual
15 minute pickle files into a single file named "data.pickle". Running
concatenate_data.py will create a data file for each month in the data directory.
It expects its working directory to be the same as \_\_main\_\_.py


Because the size of data becomes a problem over long periods of time, a compressed
data file is also made available. This is achieved by reducing all of the floats
in the data file to 16 bit floats, which were found to be accurate enough over
long periods. Compression may be undesirable for short periods of time. This
functionality can be disabled by commenting out the following lines:

    small_spectrogram = downsize_spectrogram(full_spectrogram)

    small_spectrogram.save("reduced_data.pickle")
