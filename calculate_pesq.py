#!/usr/bin/python
##-*- coding:UTF-8 -*-
import os

def calculate_pesq():
    # Calculate PESQ of all enhaced speech. 
 
    enh_speech_dir = './reconst_result'#重构后的增强语音存放的目录
    speech_dir = './SE_data/clean_wav'#干净语音存放的目录
    
    # Remove already existed file. 
    # os.system('rm _pesq_itu_results.txt')
    # os.system('rm _pesq_results.txt')
    
    # Calculate PESQ of all enhaced speech.     
    names = os.listdir(enh_speech_dir)
    for (cnt, na) in enumerate(names):
        print(cnt, na)
        enh_path = os.path.join(enh_speech_dir, na)
        #print(enh_path)
        
        speech_na = na.split('.')[0]
        speech_na = speech_na[3:]
        speech_path = os.path.join(speech_dir, "%s.wav" % speech_na)
        #print(speech_path)
        
        # Call executable PESQ tool. 
        cmd = ' '.join(["./pesq", speech_path, enh_path, "+16000"])
        #print(cmd)
        os.system(cmd)

if __name__ == "__main__":
	calculate_pesq()