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

    # build actor, input is observation(State)
    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # network structure:
            x = obs
            x = tf.cast(x, tf.float32)  # convert type to float32
            # add 3 conv
            # inputs, num_outputs, kernel_size, stride=1
            self.conv_1 = tc.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
            self.conv_2 = tc.layers.conv2d(self.conv_1, 32, 4, 2, activation_fn=tf.nn.relu)
            self.conv_3 = tc.layers.conv2d(self.conv_2, 32, 3, 1, activation_fn=tf.nn.relu)
            self.flat = tc.layers.flatten(self.conv_3)
            # add 2 dense
            self.dens_1 = tf.layers.dense(self.flat, 200)
            if self.layer_norm:
                self.dens_1_norm = tc.layers.layer_norm(self.dens_1, center=True, scale=True)
            else:
                self.dens_1_norm = self.dens_1
            self.dens_1_relu = tf.nn.relu(self.dens_1_norm)

            self.dens_2 = tf.layers.dense(self.dens_1_relu, 200)
            if self.layer_norm:
                self.dens_2_norm = tc.layers.layer_norm(self.dens_2, center=True, scale=True)
            else:
                self.dens_2_norm = self.dens_2
            self.dens_2_relu = tf.nn.relu(self.dens_2_norm)

            # output layer
            self.out = tf.layers.dense(self.dens_2_relu, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            self.out_tanh = tf.nn.tanh(self.out)  # (-1,1)

        return self.out_tanh


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
            self.conv_1 = tc.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
            self.conv_2 = tc.layers.conv2d(self.conv_1, 32, 4, 2, activation_fn=tf.nn.relu)
            self.conv_3 = tc.layers.conv2d(self.conv_2, 32, 3, 1, activation_fn=tf.nn.relu)
            self.flat = tc.layers.flatten(self.conv_3)
            #add 2 dense
            self.dens_1 = tf.layers.dense(self.flat, 200)
            if self.layer_norm:
                self.dens_1_norm = tc.layers.layer_norm(self.dens_1, center=True, scale=True)
            else:
                self.dens_1_norm = self.dens_1
            self.dens_1_relu = tf.nn.relu(self.dens_1_norm)

            self.dens_2_action = tf.concat([self.dens_1_relu, action], axis=-1)#plug action in
            self.dens_2 = tf.layers.dense(self.dens_2_action, 200)
            if self.layer_norm:
                self.dens_2_norm = tc.layers.layer_norm(self.dens_2, center=True, scale=True)
            else:
                self.dens_2_norm = self.dens_2
            self.dens_2_relu = tf.nn.relu(self.dens_2_norm)
        
            #output layer
            self.out = tf.layers.dense(self.dens_2_relu, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        
        
        return self.out

#     @property
#     def output_vars(self):
#         output_vars = [var for var in self.trainable_vars if 'output' in var.name]
#         return output_vars
