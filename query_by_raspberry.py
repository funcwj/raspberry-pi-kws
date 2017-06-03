#!/usr/bin/env python

'''QbyE Demo'''

import sys
import Queue
import threading
import pyfbank
import pynnet1
import math
import time
import kaldi_io
import numpy as np


MAX_SIZE_OF_QUEUE = 200
MAX_SIZE_OF_BUFFER = 1300
LEFT_CONTEXT = 10
RIGHT_CONTEXT = 5


class ListenThread(threading.Thread):
    '''fetch data from pipe and get filter bank feature'''
    def __init__(self, queue):
        self.fbank_queue = queue
        threading.Thread.__init__(self)
        self.running = False
        self.computer = pyfbank.fbankcomputer()

    def stop(self):
        '''given a signal to stop this thread'''
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            pcm = sys.stdin.read(MAX_SIZE_OF_BUFFER)
            if len(pcm) != MAX_SIZE_OF_BUFFER: break;
            wave_short = np.fromstring(pcm, dtype=np.int16)
            wave_float = np.array(wave_short, dtype=np.float32)
            data_feats = self.computer.compute(wave_float)
            assert data_feats.shape[1] == 40
            for idx in range(data_feats.shape[0]):
                feats_per_frame = np.zeros(40, dtype=np.float32)
                feats_per_frame = data_feats[idx, :]
                self.fbank_queue.put(feats_per_frame)
        self.fbank_queue.put(np.zeros(1))
        print "ListenThread exit..."

class ExpectThread(threading.Thread):
    '''get posteriors from the nnet1'''
    def __init__(self, queue):
        self.fbank_queue = queue
        threading.Thread.__init__(self)
        self.nnet1 = pynnet1.nnet1('final.nnet', 'templ/template.post')
        self.nnet1.debug(False)
        self.nnet1.threshold(0.5)
        self.nnet1.windowsize(8)

    def run(self):
        feats_len = 40
        totol_len = 40 * (LEFT_CONTEXT + RIGHT_CONTEXT + 1)
        nnet_in = np.zeros(totol_len, dtype=np.float32)
        feats_per_frame = self.fbank_queue.get(True)
        # init
        for idx in range(LEFT_CONTEXT + RIGHT_CONTEXT + 1):
            base = idx * feats_len
            if idx > LEFT_CONTEXT:
                feats_per_frame = self.fbank_queue.get(True)
            nnet_in[base: base + feats_len] = feats_per_frame
        # loop
        while True:
            self.nnet1.postprocess(nnet_in, 1)
            feats_per_frame = self.fbank_queue.get(True)
            # print self.fbank_queue.qsize()
            if feats_per_frame.size == 1:
                break
            nnet_in[0: totol_len - feats_len] = nnet_in[feats_len: totol_len]
            nnet_in[totol_len - feats_len: totol_len] = feats_per_frame[:]
        # print "ExpectThread exit..."

def main():
    '''logic control to handle threads'''
    fbank_queue = Queue.Queue(MAX_SIZE_OF_QUEUE)
    listen_thread = ListenThread(fbank_queue)
    expect_thread = ExpectThread(fbank_queue)
    try:
        listen_thread.start()
        expect_thread.start()
    except KeyboardInterrupt:
        listen_thread.stop()
        expect_thread.join()

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "redundant parameters..."
        sys.exit(1)
    main()
