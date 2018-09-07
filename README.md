# DDPG-by-tensorflow

The python files contain the code to set up the DDPG agent as well as the game environment.In the file called tools, we define the preprocessing approach. In jump_env, we set up the game environment of the jump game. In memory, the memory of the agent is defined. In models, the actor and critic is defined. In ddpg, we define the DDPG agent.





这里的ddpg代码实现参考了openai的baseline https://github.com/openai/baselines

这个模型目前被用来玩微信的跳一跳小游戏，电脑安装好adb，手机连接电脑并且打开开发者模式，允许USB调制，手动打开跳一跳游戏到游戏界面，运行train-ddpg-model即可看到模型成功运行。



这里我们直接截图游戏图像作为数据输入，tools中定义了用来进行图像预处理的方法；
memory中定义了模型使用的memory的类；
models中定义了actor与critic两个神经网络，这里采用的是卷积神经网络，模型结构设计参考了论文 CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING；
ddpg中定义了ddpg agent


