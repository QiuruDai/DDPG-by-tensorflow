from copy import copy
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)
def get_target_updates(vars, target_vars, tau):
    print('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    print('len',len(vars),'=',len(target_vars))
    for var, target_var in zip(vars, target_vars):
        print('{',target_var.name,'} <- {',var.name,'}')
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


class DDPG(object):
    '''
        parameters are not apply:
        normalize_observations=True, normalize_returns=False, observation_range=(-5., 5.),
        param_noise=None, enable_popart=False, critic_l2_reg=0.,clip_norm=None,
        return_range=(-np.inf, np.inf),
        '''
    def __init__(self, actor, critic, memory, observation_shape, action_shape,
                 actor_lr=1e-4, critic_lr=1e-3, decay_rate=0.9, decay_steps=5000, batch_size=128,
                 gamma=0.99, tau=0.001, action_range=(-1., 1.),
                 reward_scale=1.,action_noise=None):
        #？？？
        #adaptive_param_noise=True, adaptive_param_noise_policy_threshold=.1,
        
        #input: s0, s1, death, r, a
        self.obs0 = tf.placeholder(tf.float32, shape=(None,)+observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,)+observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        #critic target placeholder
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        
        #param_noise is not apply
        #self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        
        #1.initialise actor and critic
        self.critic = critic
        self.actor = actor
        #3.initialise memory
        self.memory = memory
        
        #set up global step
        self.global_step = tf.Variable(0, trainable=False)
        
        #set learning rate decay
        self.actor_lr = tf.train.exponential_decay(actor_lr, self.global_step, decay_steps, decay_rate, staircase=True)
        self.critic_lr = tf.train.exponential_decay(critic_lr, self.global_step, decay_steps, decay_rate, staircase=True)


        #Parameters.
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.reward_scale = reward_scale
        self.stats_sample = None
        self.action_range = action_range
        
        '''
            #parameters not used
            self.normalize_observations = normalize_observations
            self.normalize_returns = normalize_returns
            self.param_noise = param_noise
            self.return_range = return_range
            self.observation_range = observation_range
            self.clip_norm = clip_norm
            self.enable_popart = enable_popart
            self.critic_l2_reg = critic_l2_reg
            
            
            # Observation normalization did not apply
            # Return normalization did not apply
            '''
        # Create target networks
        #2.initialize target networks
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic
        
        
        # Create networks and core TF parts that are shared across setup parts. normalized_obs0 & 1 is replaced by self.obs0 & 1 since normalization is not applied
        #estimated action0 by actor
        self.actor_tf = actor(self.obs0)
        #time0: estimated-Q0 with the actual action from sampled memory
        self.critic_tf = critic(self.obs0, self.actions)
        #time0: estimated-Q0 with the action from actor 用于更新actor
        #???reuse
        self.critic_with_actor_tf = critic(self.obs0, self.actor_tf, reuse=True)
        #time1: target-Q0
        #estimated-Q1 by target critic at s1 with the predict future action from target actor
        Q_obs1 = target_critic(self.obs1, target_actor(self.obs1))
        self.target_Q = self.rewards + (1. - self.terminals1) * self.gamma * Q_obs1
        
        
        # Set up parts.
        self.setup_target_network_updates()
        #param_noise is not applied, setup_param_noise is not defined
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        #train ops???
        self.train_ops = [self.actor_loss, self.critic_loss, self.actor_train, self.critic_train]
        #popart is not applied
        self.setup_stats()
    #         self.setup_target_network_updates()
    
    
    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]
    
    #def setup_param_noise(self, normalized_obs0):

    def setup_actor_optimizer(self):
        #???logger.info
        print('setting up actor optimizer')
        # - (how much Q our agent could get)
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        #actor_shapes & actor_nb_params are not reported
        #functions from baseline are not used, we use tf to calculate grad and optimise
        #self.actor_grads =
        # self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.actor_train = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, global_step = self.global_step, var_list=self.actor.trainable_vars)
    
    
    def setup_critic_optimizer(self):
        #???
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_tf - self.critic_target))
        #critic_l2_reg is not applied
        #critic_shapes & critic_nb_params are not reported
        # self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)
        self.critic_train = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, var_list=self.critic.trainable_vars)
    
    #def setup_popart(self):
    
    def setup_stats(self):
        #things not applied are not recorded
        ops = []
        names = []
        
        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']
        
        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']
        
        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']
        
        self.stats_ops = ops
        self.stats_names = names
    
    #key: apply_noise is not used
    #pi is used to compute a and q given obs???
    def pi(self, obs, apply_noise=True, compute_Q=True):
        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [obs]}
        #???compute_Q
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        #flatten()???
        action = action.flatten()
        
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise

        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q
    
    def store_transition(self, obs0, action, reward, obs1, terminal1):
        #reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)
    
    def train(self):
        #get a batch
        batch = self.memory.sample(batch_size=self.batch_size)
        #compute target_Q
        target_Q = self.sess.run(self.target_Q, feed_dict={
                                 self.obs1: batch['obs1'],
                                 self.rewards: batch['rewards'],
                                 self.terminals1: batch['terminals1'].astype('float32'),
                                 })
        #compute loss
        actor_loss, critic_loss, _, _ = self.sess.run(self.train_ops, feed_dict={
                                                      self.obs0: batch['obs0'],
                                                      self.actions: batch['actions'],
                                                      self.critic_target: target_Q,
                                                      })
                                 
                                 
        return critic_loss, actor_loss
    
    def initialize(self, sess):
        #get sess here
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)
    
    def update_target_net(self):
        self.sess.run(self.target_soft_updates)
    
    def get_stats(self):
        #与setup_stats配合使用
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
                               self.obs0: self.stats_sample['obs0'],
                               self.actions: self.stats_sample['actions'],
                               })
        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))
                               
        return stats


    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
