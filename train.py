import os
import time
import numpy as np
import tensorflow as tf

import cv2
from jump_env import Jump_Env
from tools import state
from models import Actor, Critic
from memory import Memory
from ddpg import DDPG


#this function only train model on data, do not load model
def train_on_data(env, steps, data, experiment_dir,
                  actor, critic, memory,
                  actor_lr, critic_lr, batch_size,
                  gamma, tau=0.01):
    
    #build agent: action_range=(-1., 1.),reward_scale=1.
    agent = DDPG(actor, critic, memory, env.observation_shape, env.action_shape,
                 actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size,
                 gamma=gamma, tau=tau)
    
    #put data into memory
    print('Loading memory...')
    for i in range(len(data)):
        obs = state(data.iat[i,0])
        action = env.presstime_to_action(data.iat[i,1])
        r = data.iat[i,3]
        new_obs = state(data.iat[i,2])
        done = data.iat[i,4]
        agent.store_transition(obs, action, r, new_obs, done)

    #saver
    saver = tf.train.Saver()
    #------add save dir--------
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #summary dir
    summary_dir = os.path.join(experiment_dir, "summaries")
    if not os.path.exists(summary_dir):#如果路径不存在创建路径
        os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary = tf.Summary()
    #----------------------------
    with tf.Session() as sess:
        
        # Prepare everything.
        agent.initialize(sess)
        #         sess.graph.finalize()
        
        #------------------------
        print('Training...')
        for step in range(steps):
            
            t0 = time.time()
            
            # Train.
            cl, al = agent.train()
            
            t1 = time.time()
            tt = t1-t0
            
            #record loss
            summary.value.add(simple_value=cl, tag="critic_loss")
            summary.value.add(simple_value=al, tag="actor_loss")
            summary.value.add(simple_value=tt, tag="train_time")
            summary_writer.add_summary(summary, step)
            
            #record graph
            summary_writer.add_graph(sess.graph)
            
            #flush
            summary_writer.flush()
            
            #update model
            agent.update_target_net()
            
            #save model every 100 steps
            if step%100 == 0:
                saver.save(tf.get_default_session(), checkpoint_path)

    print('Training completed.')


#this function only train model on data
#load model if we have one, load data if there are more data
def train_on_data_online(env, steps, data, experiment_dir,
                         actor, critic, memory,
                         actor_lr, critic_lr, batch_size,
                         gamma, tau=0.01):
    
    #build agent: action_range=(-1., 1.),reward_scale=1.
    agent = DDPG(actor, critic, memory, env.observation_shape, env.action_shape,
                 actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size,
                 gamma=gamma, tau=tau)
    #put data into memory
    init_data = 1000
    print('Loading ',init_data,' memory...')
    assert len(data)>=init_data
    for i in range(init_data):
        obs = state(data.iat[i,0])
        action = env.presstime_to_action(data.iat[i,1])
        r = data.iat[i,3]
        new_obs = state(data.iat[i,2])
        done = data.iat[i,4]
        agent.store_transition(obs, action, r, new_obs, done)

    #saver
    saver = tf.train.Saver()
    #------add save dir--------
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #summary dir
    summary_dir = os.path.join(experiment_dir, "summaries")
    if not os.path.exists(summary_dir):#如果路径不存在创建路径
        os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary = tf.Summary()
    #----------------------------
    with tf.Session() as sess:
        
        #load model if we have
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            agent.sess = sess
        else:
        # Prepare everything.
            print('Building new model...')
            agent.initialize(sess)
        #         sess.graph.finalize()
        
        #------------------------
        print('Training...')
        data_inx = init_data
        
        for step in range(steps):
            #load 1 data if there are more data
            if data_inx < len(data):
                obs = state(data.iat[data_inx,0])
                action = env.presstime_to_action(data.iat[data_inx,1])
                r = data.iat[data_inx,3]
                new_obs = state(data.iat[data_inx,2])
                done = data.iat[data_inx,4]
                agent.store_transition(obs, action, r, new_obs, done)
                data_inx += 1
        
            # Train.
            cl, al = agent.train()
            global_step = sess.run(agent.global_step)
            #record loss
            summary.value.add(simple_value=cl, tag="critic_loss")
            summary.value.add(simple_value=al, tag="actor_loss")
            summary_writer.add_summary(summary, global_step)
            
            #             #record graph
            #             summary_writer.add_graph(sess.graph)
            
            #flush
            summary_writer.flush()
            
            #update model
            agent.update_target_net()
            
            #save model every 1000 steps
            if step%1000 == 0:
                saver.save(tf.get_default_session(), checkpoint_path)

    print('Training completed!')


# 在数据训练的模型的基础上继续跳着训练,即标准ddpg方法边做动作边收集数据边训练
# load model if we have one, put initial data if we have
def train_jump_after_data(env, episodes, data, experiment_dir,
                          actor, critic, memory,
                          actor_lr, critic_lr, batch_size,
                          gamma, tau=0.01):
    # build agent: action_range=(-1., 1.),reward_scale=1.
    agent = DDPG(actor, critic, memory, env.observation_shape, env.action_shape,
                 actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size,
                 gamma=gamma, tau=tau)
    # put data into memory
    print('Loading ', len(data), ' memory...')
    for i in range(len(data)):
        obs = state(data.iat[i, 0])
        action = env.presstime_to_action(data.iat[i, 1])
        r = data.iat[i, 3]
        new_obs = state(data.iat[i, 2])
        done = data.iat[i, 4]
        agent.store_transition(obs, action, r, new_obs, done)

    # saver
    saver = tf.train.Saver()
    # ------add save dir--------
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # summary dir
    summary_dir = os.path.join(experiment_dir, "summaries")
    if not os.path.exists(summary_dir):  # 如果路径不存在创建路径
        os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary = tf.Summary()
    episode_summary = tf.Summary()
    # ----------------------------
    with tf.Session() as sess:

        # load model if we have
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            agent.sess = sess
        else:
            # Prepare everything.
            print('Building new model...')
            agent.initialize(sess)
        #         sess.graph.finalize()

        # ------------------------
        print('Training...')

        for episode in range(episodes):
            # set game
            #             print('new game')
            obs0 = env.reset()
            episode_reward = 0
            episode_step = 0

            while 1:

                # Train.
                cl, al = agent.train()
                global_step = sess.run(agent.global_step)
                # record loss
                summary.value.add(simple_value=cl, tag="critic_loss")
                summary.value.add(simple_value=al, tag="actor_loss")
                summary_writer.add_summary(summary, global_step)

                #             #record graph
                #             summary_writer.add_graph(sess.graph)

                # flush
                summary_writer.flush()

                # update model
                agent.update_target_net()

                # -----------------------------------
                # get action
                feed_dict = {agent.obs0: [obs0]}
                action = sess.run(agent.actor_tf, feed_dict=feed_dict)
                action = action.flatten()

                # do action
                obs1, reward, done = env.step(action)
                episode_reward += reward
                episode_step += 1

                # store transition
                agent.store_transition(obs0, action, reward, obs1, done)
                obs0 = obs1

                if done:
                    episode_summary.value.add(simple_value=episode_reward, tag="episode_reward")
                    episode_summary.value.add(simple_value=episode_step, tag="episode_step")
                    summary_writer.add_summary(episode_summary, episode)
                    summary_writer.flush()
                    #                     print('dead at',episode_step)
                    break

                # ----------------------------------------------------------

            # save model every 100 episodes
            if episode % 100 == 0:
                saver.save(tf.get_default_session(), checkpoint_path)

    print('Training completed!')


# train while playing game, we do not need any data
def train_jump(env, episodes, init_memory, experiment_dir,
               actor, critic, memory,
               actor_lr, critic_lr, batch_size,
               gamma, tau=0.01):
    # build agent: action_range=(-1., 1.),reward_scale=1.
    agent = DDPG(actor, critic, memory, env.observation_shape, env.action_shape,
                 actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size,
                 gamma=gamma, tau=tau)

    # saver
    saver = tf.train.Saver()
    # ------add save dir--------
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # summary dir
    summary_dir = os.path.join(experiment_dir, "summaries")
    if not os.path.exists(summary_dir):  # 如果路径不存在创建路径
        os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary = tf.Summary()
    episode_summary = tf.Summary()
    # ----------------------------
    with tf.Session() as sess:

        # load model if we have
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            agent.sess = sess
        else:
            # Prepare everything.
            print('Building new model...')
            agent.initialize(sess)
        #         sess.graph.finalize()

        # ------------------------
        # generate initial memory
        print('Generating ', init_memory, ' memory... Please reset game!')
        obs0 = env.reset()
        for i in range(init_memory):
            #             set game
            print('new game')
            obs0 = env.reset()
            while 1:
                # get action
                feed_dict = {agent.obs0: [obs0]}
                action = sess.run(agent.actor_tf, feed_dict=feed_dict)
                action = action.flatten()

                # do action
                obs1, reward, done = env.step(action)

                # store transition
                agent.store_transition(obs0, action, reward, obs1, done)

                # judge death
                if done:
                    break
                else:
                    obs0 = obs1
        # ---------------------------------------

        print('Training...')
        for episode in range(episodes):
            # set game
            #             print('new game')
            obs0 = env.reset()
            episode_reward = 0
            episode_step = 0

            while 1:

                # Train.
                cl, al = agent.train()
                global_step = sess.run(agent.global_step)
                # record loss
                summary.value.add(simple_value=cl, tag="critic_loss")
                summary.value.add(simple_value=al, tag="actor_loss")
                summary_writer.add_summary(summary, global_step)

                #             #record graph
                #             summary_writer.add_graph(sess.graph)

                # flush
                summary_writer.flush()

                # update model
                agent.update_target_net()

                # -----------------------------------
                # get action
                feed_dict = {agent.obs0: [obs0]}
                action = sess.run(agent.actor_tf, feed_dict=feed_dict)
                action = action.flatten()

                # do action
                obs1, reward, done = env.step(action)
                episode_reward += reward
                episode_step += 1

                # store transition
                agent.store_transition(obs0, action, reward, obs1, done)
                obs0 = obs1

                if done:
                    episode_summary.value.add(simple_value=episode_reward, tag="episode_reward")
                    episode_summary.value.add(simple_value=episode_step, tag="episode_step")
                    summary_writer.add_summary(episode_summary, episode)
                    summary_writer.flush()
                    #                     print('dead at',episode_step)
                    break

                # ----------------------------------------------------------

            # save model every 100 episodes
            if episode % 100 == 0:
                saver.save(tf.get_default_session(), checkpoint_path)

    print('Training completed!')

if __name__ == '__main__':
    # hyper-parameters of training on data
    actor_lr = 1e-4
    critic_lr = 1e-3
    batch_size = 64
    gamma = 0.99
    tau = 0.01
    nb_actions = 1
    limit = int(5000)
    init_memory = 100
    episodes = 10000

    #save dir
    experiment_dir = os.path.abspath("./ddpg-model/experiments/")

    #build env
    number_templet = [cv2.imread('templet/{}.jpg'.format(i)) for i in range(10)]
    restart_templet = cv2.imread('templet/again.jpg')
    env = Jump_Env(number_templet=number_templet, restart_templet=restart_templet)

    #build elements
    actor = Actor(nb_actions, layer_norm=True)
    critic = Critic(layer_norm=True)
    memory = Memory(limit, action_shape=env.action_shape, observation_shape=env.observation_shape)

    #train, you can switch to other train function to train in different ways
    train_jump(env=env, episodes=episodes, init_memory=init_memory, experiment_dir=experiment_dir, actor=actor,
               critic=critic, memory=memory,
               actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, gamma=gamma, tau=tau)






