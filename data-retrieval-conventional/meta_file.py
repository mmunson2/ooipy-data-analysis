import pickle


class MetaFile:
    META_FILE_NAME = 'meta'
    PICKLE_EXTENSION = '.pickle'
    TEXT_EXTENSION = '.txt'

    def __init__(self, start_time, end_time, segment_length, segment_count,
                 dir_name, args):
        self.dir_name = dir_name
        self.start_time = start_time
        self.end_time = end_time
        self.segment_length = segment_length
        self.segment_count = segment_count
        self.missed_segments = None

        self.node = args.get('node')
        self.fmin = args.get('fmin')
        self.fmax = args.get('fmax')

        self.win = args.get('win')
        self.L = args.get('L')
        self.overlap = args.get('overlap')
        self.avg_method = args.get('avg_method')
        self.interpolate = args.get('interpolate')
        self.scale = args.get('scale')

    def set_missed_segments(self, missed_segments):
        self.missed_segments = missed_segments

    def save_meta_data(self):
        self.__write_text_file()
        self.__write_pickle_file()

    def __write_text_file(self):
        meta = open(self.META_FILE_NAME + self.TEXT_EXTENSION, "w")

        meta.write("__________________________________________________________"
                   "______________________\n")
        meta.write(self.dir_name + " Metadata" + "\n")
        meta.write("\n")
        meta.write("Start Time: " + self.__to_string(self.start_time) + "\n")
        meta.write("End Time: " + self.__to_string(self.end_time) + "\n")

        elapsed_time = self.end_time - self.start_time

        # Todo: Add units to all files
        meta.write("Elapsed Length: " + self.__to_string(
            elapsed_time) + " (hh:mm:ss)" + "\n")
        meta.write("\n")

        meta.write("Segment Length: " + self.__to_string(
            self.segment_length) + " minutes" + "\n")
        meta.write("Segment Count: " +
                   self.__to_string(self.segment_count) + " segments" + "\n")
        meta.write("\n")

        if self.missed_segments is None:
            meta.write("Missed Segments: missed segment count not recorded!")
        else:
            meta.write("Missed Segments: " + self.__to_string(
                self.missed_segments) + " segments" + "\n")

        meta.write("\n")

        meta.write("__________________________________________________________"
                   "______________________\n")
        meta.write("Data Retrieval Settings:\n")
        meta.write("\n")

        meta.write("Node: " + self.__to_string(self.node) + "\n")
        meta.write("Node Location: " + get_node_location(self.node) + "\n")
        meta.write("\n")

        meta.write(
            "Minimum Frequency: " + self.__to_string(self.fmin) + " Hz" + "\n")
        meta.write(
            "Maximum Frequency: " + self.__to_string(self.fmax) + " Hz" + "\n")
        meta.write("\n")

        meta.write("__________________________________________________________"
                   "______________________\n")
        meta.write("Power Spectral Density Calculation Settings:\n")
        meta.write("\n")

        meta.write("Window Function: " + self.__to_string(self.win) + "\n\n")
        meta.write("Data Block Length: " + self.__to_string(self.L) + "\n\n")
        meta.write(
            "Overlap Percentage: " + self.__to_string(self.overlap) + "\n\n")
        meta.write(
            "Averaging Method: " + self.__to_string(self.avg_method) + "\n\n")
        meta.write("Interpolation Method: " + self.__to_string(
            self.interpolate) + "\n\n")
        meta.write("Scale: " + self.__to_string(self.scale) + "\n\n")

        meta.write("__________________________________________________________"
                   "______________________\n")
        meta.write("Notes:")
        meta.write("\n")

        meta.close()

    def __write_pickle_file(self):
        with open(self.META_FILE_NAME + self.PICKLE_EXTENSION,
                  'wb') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def __to_string(val):

        if val is None:
            return "None"
        else:
            return str(val)


def get_node_location(node):
    if node == 'LJ01D':
        return "Oregon Shelf Base Seafloor"
    if node == 'LJ01A':
        return "Oregon Slope Base Seafloor"
    if node == "PC01A":
        return "Oregon Slope Base Shallow"
    if node == "PC03A":
        return "Axial Base Shallow Profiler"
    if node == "LJ01C":
        return "Oregon Offshore Base Seafloor"
    if node == "LJ03A":
        return "Axial Base Seafloor"
    else:
        return "Unregistered Location"
