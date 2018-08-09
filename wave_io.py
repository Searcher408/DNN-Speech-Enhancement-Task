#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#by aslp-wujian@2017.4.15
# 对wav文件读取进行了封装，调用next_frame_phase方法返回的当前帧的相位信息

import wave
import numpy as np


class WaveWrapper(object):

    def __init__(self, path, time_wnd = 25, time_off = 6.25):
        wave_src = wave.open(path, "rb")
        para_src = wave_src.getparams()
        self.rate = int(para_src[2]) 
        self.cur_size = 0
        self.tot_size = int(para_src[3])
        # default 400 100
        self.wnd_size = int(self.rate * 0.001 * time_wnd)
        self.wnd_rate = int(self.rate * 0.001 * time_off)
        self.ham = np.hamming(self.wnd_size)
        self.data = np.fromstring(wave_src.readframes(wave_src.getnframes()), dtype=np.int16)
        self.upper_bound = np.max(np.abs(self.data))

    def get_frames_num(self):
        return int((self.tot_size - self.wnd_size) / self.wnd_rate + 1)

    def get_wnd_size(self):
        return self.wnd_size

    def get_wnd_rate(self):
        return self.wnd_rate

    def get_sample_rate(self):
        return self.rate

    def get_upper_bound(self):
        return self.upper_bound

    def next_frame_phase(self):
        while self.cur_size + self.wnd_size <= self.tot_size:
            value = np.zeros(512)
            value[: self.wnd_size] = np.array(self.data[self.cur_size: \
                    self.cur_size + self.wnd_size], dtype=np.float)
            value -= np.sum(value) / self.wnd_size
            value[: self.wnd_size] *= self.ham
            angle = np.angle(np.fft.rfft(value))
            yield np.cos(angle) + np.sin(angle) * 1.0j
            self.cur_size += self.wnd_rate
