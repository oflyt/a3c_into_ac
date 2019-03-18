import threading

class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self, brain, exception_bucket):
        threading.Thread.__init__(self)
        self.brain = brain
        self.exception_bucket = exception_bucket

    def run(self):
        """Main loop for thread
        
        Tells the brain to optimize the neural network
        """
        try:
            while not self.stop_signal:
                self.brain.optimize()
        except Exception as e:
            self.exception_bucket.put(e)

    def stop(self):
        "Send stop signal to thread"
        self.stop_signal = True