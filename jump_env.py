import os
import cv2
import time
import random
from operator import itemgetter
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
# import tensorflow.contrib as tc

from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

from collections import deque# Ordered collection with ends
from tools import state



class Data_Env(object):
    def __init__(self):
        self.observation_shape = (60,60)
        self.action_shape = (1,)

    #action (-1,1) presstime (300,1500)
    def action_to_presstime(self, action):
        assert -1<=action and action<=1
        presstime = (action + 1) * 600 + 300
        assert 300<= presstime and presstime<=1500
        presstime = int(presstime)
        return presstime
    def presstime_to_action(self, presstime):
        assert 300<= presstime and presstime<=1500
        action = (presstime - 300) / 600 - 1
        assert -1<=action and action<=1
        return action
    





class Jump_Env(object):
    def __init__(self, number_templet, restart_templet, threshold = 0.95):
        self.observation_shape = (60,60)
        self.action_shape = (1,)
        self.number_templet = number_templet
        self.restart_templet = restart_templet
        self.threshold = threshold#for templet match
    
    #return current state
    def reset(self):
        #游戏不能指定最大步数结束，只能自然死亡再来一局
#        print('Please reset the game!!!')
        #get state
        self.pull_screenshot('autojump.jpg')
        obs = state('autojump.jpg')
        return obs
    
    
    
    #     def step(self, action):
    #         #new_obs, r, done = env.step(action)
    #         pass
    
    #reward=1 if not dead
    def step(self, action):
        #do action
        press_time = self.action_to_presstime(action)
        self.jump(press_time)
#        print('action:',action)
#        print('press_time',press_time)
        time.sleep(3.9)
        
        #get state
        self.pull_screenshot('autojump.jpg')
        obs = state('autojump.jpg')
        
        # Game Over
        if self.restart('autojump.jpg'):
            done = 1
            reward = -1
        else:
            done = 0
            reward = 1
        
        return obs, reward, done
    
    #action (-1,1) presstime (300,1500)
    def action_to_presstime(self, action):
        assert -1<=action and action<=1
        presstime = (action + 1) * 600 + 300
        assert 300<= presstime and presstime<=1500
        presstime = int(presstime)
        return presstime
    def presstime_to_action(self, presstime):
        assert 300<= presstime and presstime<=1500
        action = (presstime - 300) / 600 - 1
        assert -1<=action and action<=1
        return action
    
    def pull_screenshot(self, file_name):
        # Get screenshot
        os.system('adb shell screencap -p /sdcard/%s' % file_name)
        os.system('adb pull /sdcard/%s .' % file_name)
    
    def jump(self, press_time):
        rand = random.randint(0, 9) * 10
        cmd = ('adb shell input swipe %i %i %i %i ' + str(press_time)) \
            % (320 + rand, 410 + rand, 320 + rand, 410 + rand)
        os.system(cmd)
    #print(cmd)
    
    #输入按压时间和位置，发出adb命令，restart
    def start(self, press_time, swipe_x1, swipe_y1, swipe_x2, swipe_y2):
        press_time = int(press_time)
        cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
                                                                            x1=swipe_x1,
                                                                            y1=swipe_y1,
                                                                            x2=swipe_x2,
                                                                            y2=swipe_y2,
                                                                            duration=press_time)
        os.system(cmd)
    
    #分数
    def get_score(self, file_name):
        # Get score from image
        
        background = cv2.imread(file_name)
        
        match_result = []
        for i, number in enumerate(self.number_templet):
            res = cv2.matchTemplate(number, background, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res > self.threshold)
            for pt in zip(*loc[::-1]):
                match_result.append((i, pt[0]))
        match_result.sort(key=itemgetter(1))
        
        score = 0
        last_position = 0
        for x in match_result:
            if x[1] - last_position < 30:
                continue
            score = 10 * score + x[0]
            last_position = x[1]

        return score
    
    def restart(self, file_name):
        # Check game over and restart
        h, w, _ = self.restart_templet.shape
        background = cv2.imread(file_name)
        res = cv2.matchTemplate(self.restart_templet, background, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > self.threshold:
            top_left = max_loc
            left, top = (top_left[0] + w / 2, top_left[1] + h / 2)
            self.start(100, left, top, left, top)
            time.sleep(3.9)
            return True
        else:
            return False


#def main():
#   env = Jump_Env(number_templet,restart_templet,0.95)
#   env.pull_screenshot("test.jpg")
#   print(env.restart("test.jpg"))
#   # print(env.get_score("test.jpg"))
#
#if __name__ == '__main__':
#    number_templet = [cv2.imread('templet/{}.jpg'.format(i)) for i in range(10)]
#    restart_templet = cv2.imread('templet/again.jpg')
#    main()
