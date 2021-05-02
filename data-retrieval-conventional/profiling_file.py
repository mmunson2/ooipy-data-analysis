
import pickle
from datetime import datetime
import time



class ProfilingFile:
    PROFILING_FILE_NAME = 'profile'
    PICKLE_EXTENSION = '.pickle'
    TEXT_EXTENSION = '.txt'

    def __init__(self, dir_name):
        self.dir_name = dir_name

        self.start_time = datetime.utcnow()
        self.end_time = None

        self.compute_times = []
        self.retrieval_times = []

        self.__retrieval_start = None
        self.__computation_start = None

    def save_profiling_data(self):
        self.__write_text_file()
        self.__write_pickle_file()

    def start_data_retrieval(self):
        self.__retrieval_start = time.time()

    def end_data_retrieval(self):
        self.retrieval_times.append(time.time() - self.__retrieval_start)

    def start_computation(self):
        self.__computation_start = time.time()

    def end_computation(self):
        self.compute_times.append(time.time() - self.__computation_start)

    def __write_text_file(self):
        perf = open(self.PROFILING_FILE_NAME + self.TEXT_EXTENSION, "w")

        perf.write("__________________________________________________________"
                   "______________________\n")
        perf.write(self.dir_name + " Performance Summary" + "\n")
        perf.write("\n")

        perf.write("Job Start Time: " + self.__to_string(self.start_time)
                   + "\n")

        if self.end_time is None:
            self.end_time = datetime.utcnow()

        perf.write("Job End Time: " + self.__to_string(self.end_time) + "\n")

        elapsed_time = self.end_time - self.start_time

        perf.write("Elapsed Time: " + self.__to_string(elapsed_time) + "\n")
        perf.write("\n")

        if len(self.retrieval_times) > 0:
            combined_seconds = 0

            for timespan in self.retrieval_times:
                combined_seconds += timespan

            retrieve_average = self.__to_string(
                combined_seconds / len(self.retrieval_times)) + " seconds"
        else:
            retrieve_average = "No retrieval times available"

        if len(self.compute_times) > 0:
            combined_seconds = 0

            for timespan in self.compute_times:
                combined_seconds += timespan

            compute_average = self.__to_string(
                combined_seconds / len(self.compute_times)) + " seconds"
        else:
            compute_average = "No compute times available"

        perf.write("Average Retrieval Time: " + retrieve_average + "\n")
        perf.write("Average Compute Time: " + compute_average + "\n")
        perf.write("\n")

        perf.write("__________________________________________________________"
                   "______________________\n")
        perf.write("Notes:")
        perf.write("\n")

        perf.close()

    def __write_pickle_file(self):
        with open(self.PROFILING_FILE_NAME + self.PICKLE_EXTENSION,
                  'wb') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def __to_string(val):

        if val is None:
            return "None"
        else:
            return str(val)
