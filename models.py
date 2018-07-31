import tensorflow as tf
import tensorflow.contrib as tc

class Model(object):
    def __init__(self, name):
        self.name = name
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    
    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

#     @property
#     def perturbable_vars(self):
#         return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


#actor
class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
    
    #build actor, input is observation(State)
    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            #network structure:
            x = obs
            x = tf.cast(x, tf.float32)#convert type to float32
            #add 3 conv
            x = tc.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
            x = tc.layers.conv2d(x, 32, 4, 2, activation_fn=tf.nn.relu)
            x = tc.layers.conv2d(x, 32, 3, 1, activation_fn=tf.nn.relu)
            x = tc.layers.flatten(x)
            #add 2 dense
            x = tf.layers.dense(x, 200)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 200)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            #output layer
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)#(-1,1)
        return x



#critic
class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
    
    #build critic, input is obs and actions
    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            #network structure
            x = obs
            x = tf.cast(x, tf.float32)
            #add 3 conv
            x = tc.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
            x = tc.layers.conv2d(x, 32, 4, 2, activation_fn=tf.nn.relu)
            x = tc.layers.conv2d(x, 32, 3, 1, activation_fn=tf.nn.relu)
            x = tc.layers.flatten(x)
            #add 2 dense
            x = tf.layers.dense(x, 200)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.concat([x, action], axis=-1)#plug action in
            x = tf.layers.dense(x, 200)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
        
            #output layer
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        
        
        return x

#     @property
#     def output_vars(self):
#         output_vars = [var for var in self.trainable_vars if 'output' in var.name]
#         return output_vars
