# DDPG-by-tensorflow

这里的ddpg代码实现参考了openai的baseline https://github.com/openai/baselines

这里我们直接截图游戏图像作为数据输入，tools中定义了用来进行图像预处理的方法
memory中定义了模型使用的memory的类
models中定义了actor与critic两个神经网络，这里采用的是卷积神经网络，模型结构设计参考了论文 CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING
ddpg中定义了ddpg agent
