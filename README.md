# DDPG-by-tensorflow

这里的ddpg代码实现参考了openai的baseline https://github.com/openai/baselines

这个模型目前被用来玩微信的跳一跳小游戏，电脑安装好adb，手机连接电脑并且打开开发者模式，允许USB调制，手动打开跳一跳游戏到游戏界面，运行train.py即可看到模型成功运行。


【train的部分暂时不可用】

这里我们直接截图游戏图像作为数据输入，tools中定义了用来进行图像预处理的方法；
memory中定义了模型使用的memory的类；
models中定义了actor与critic两个神经网络，这里采用的是卷积神经网络，模型结构设计参考了论文 CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING；
ddpg中定义了ddpg agent；
train中定义了几种训练方法仅有细微区别，你可以任意选择。
