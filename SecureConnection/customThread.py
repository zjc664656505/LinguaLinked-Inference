import threading
import time
import server

class CustomThread(threading.Thread):
    def __init__(self, thread_id, name, lock=None):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.counter = {"A":1, "B":2}

    def setName(self, name):
        self.name = name

    def setCounter(self, counter):
        self.counter["A"] = counter
        self.counter["C"] = 3

    def run(self):
        print("Starting " + self.name)
        # while True:
        #     if self.counter < 0:
        #         print("Thread " + self.name + " counting down: " + str(self.counter))
        #         break
        #     else:
        #         time.sleep(1)

        server.test(self.counter)

        print("Ending " + self.name)