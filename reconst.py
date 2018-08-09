#!/usr/bin/python
#-*- coding:UTF-8 -*-
"""transform amplitude spectrum to waveform"""
#测试clean_wav目录下的300条语音对应的noisy_for_reconst目录下的带躁语音的语音增强效果
#便于用PESQ进行测试比对
#测试语音对应的tfrecords导入训练后的DNN模型得到的幅度谱，保存在.mat格式文件中

import numpy as np
import scipy.io as scio
import wave
import wave_io
import os

def process_file_list(file_list):
	fid = open(file_list, 'r')
	proc_file_list = []
	lines = fid.readlines()
	for line in lines:
	    proc_file_list.append(line.rstrip('\n'))
	return proc_file_list

def transform():
	list_path = 'clean_wav.lst'
	file_list = process_file_list(list_path)
	for i in range(300):
		filepath = file_list[i]
		filename = os.path.basename(filepath)
		(name, _) = os.path.splitext(filename)
		print(name)
		print('count:',i)
		
		spectrum = './mat_result/'+name+'.mat'
		noisy_for_reconst = './SE_data/noisy_for_reconst/'+name+'.wav'
		reconst = './reconst_result/SE_'+ name +'.wav'

		WAVE_WARPPER = wave_io.WaveWrapper(noisy_for_reconst)
		WAVE_RECONST = wave.open(reconst, "wb")

		WND_SIZE = WAVE_WARPPER.get_wnd_size()#400
		WND_RATE = WAVE_WARPPER.get_wnd_rate()#100

		REAL_IFFT = np.fft.irfft
		HAM_WND = np.hamming(WND_SIZE)

		data = scio.loadmat(spectrum)
		print (data[name].shape)

		SPECT_ENHANCE = data[name]
		SPECT_ROWS, SPECT_COLS = SPECT_ENHANCE.shape
		print('get_frames_num()',WAVE_WARPPER.get_frames_num())

		INDEX = 0
		SPECT = np.zeros(SPECT_COLS)
		RECONST_POOL = np.zeros((SPECT_ROWS - 1) * WND_RATE + WND_SIZE)

		for phase in WAVE_WARPPER.next_frame_phase():
		    SPECT[1:] = SPECT_ENHANCE[INDEX][1: ]
		    RECONST_POOL[INDEX * WND_RATE: INDEX * WND_RATE + WND_SIZE] += \
		                REAL_IFFT(SPECT * phase)[: WND_SIZE] * HAM_WND
		    INDEX += 1
		RECONST_POOL = RECONST_POOL / np.max(np.abs(RECONST_POOL)) * WAVE_WARPPER.get_upper_bound()

		WAVE_RECONST.setnchannels(1)
		WAVE_RECONST.setnframes(RECONST_POOL.size)
		WAVE_RECONST.setsampwidth(2)
		WAVE_RECONST.setframerate(WAVE_WARPPER.get_sample_rate())
		WAVE_RECONST.writeframes(np.array(RECONST_POOL, dtype=np.int16).tostring())
		WAVE_RECONST.close()

if __name__ == "__main__":
	transform()
