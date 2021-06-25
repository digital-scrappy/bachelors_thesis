from collections import defaultdict
from statistics import  mean
import multiprocessing as mp
import pynvml
import time

class Worker(mp.Process):
    def __init__(self,queue, gpu_id):
        super(Worker, self).__init__()
        self.gpu_id = gpu_id
        self.handle = None
        self.queue = queue
        self.n = 0
        self.accumulated = 0
        self.average = 0
        # print("STARTED WATTAGE TRACKING")


    def run(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        while True:

            self.n += 1
            self.accumulated += pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self.average = self.accumulated / self.n
            self.queue.put(self.average)
            time.sleep(0.2)

class epoch_performance:

    def __init__(self, n_batches, prof_metrics_to_aggregate, gpu_id = 0):

        self.n_batches = n_batches
        self.gpu_id = gpu_id


        self.prof_readings = defaultdict(list)
        self.prof_average = defaultdict(int)
        self.metrics_to_aggregate = prof_metrics_to_aggregate

        self.queue = mp.Queue()
        self.worker = Worker(self.queue, self.gpu_id)
        self.wattage = None

        self.worker.start()

    def set_n_batches(self, n_batches):
        self.n_batches = n_batches

    def stop_wattage_worker(self):
        self.wattage = self.queue.get()
        self.worker.kill()
        time.sleep(2)
        self.worker.close()
        pass


    def report_prof(self, prof_reading):
        for i in prof_reading:
            for x in self.metrics_to_aggregate:
                self.prof_readings[x].append(vars(i)[x])



        # for i in self.prof_readings.keys():
        #     print(f"{i} batch_mean: {mean(self.prof_readings[i]) / 3 } ms")
        #     print(f"{i} epoch_estimation: {mean(self.prof_readings[i] ) /3 * self.n_batches}")
        # print(f"Wattage : {self.queue.get()}")



        pass

    def epoch_evaluation(self):
        average_gpu_time = (sum(self.prof_readings["cuda_time_total"])/ 3)/ self.n_batches
        print(f"GPU time : {average_gpu_time}\nGPU watts : {self.wattage}")
        # print(f"GPU time : {average_gpu_time}")
        return average_gpu_time * self.wattage
