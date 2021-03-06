import threading, time, os

import numpy as np
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras import backend as K

class Brain:
    MIN_BATCH = 32
    LEARNING_RATE = 5e-3

    LOSS_V = .5            # v loss coefficient
    LOSS_ENTROPY = .01     # entropy coefficient
    
    GAMMA = 0.99
    N_STEP_RETURN = 8
    GAMMA_N = GAMMA ** N_STEP_RETURN
    
    train_queue = [ [], [], [], [], [] ]    # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    
    def __init__(self, n_state, n_actions, model_weights=None):
        self.n_state = n_state
        self.n_actions = n_actions
        self.none_state = np.zeros(n_state)
        
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
             
        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
        
        if model_weights is not None:
            if not os.path.isfile(model_weights):
                print("Could not find weights file")
            else:
                self.model.load_weights(model_weights)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()    # avoid modifications
        
    def _build_model(self):
        """Build the neural network
        
        The neural network will be used to predict the policy, actor
        And to predict the value of an action given a state, critic
        """
        
        l_input = Input(batch_shape=(None,) + self.n_state)
        
        l_conv_1 = Conv2D(16,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu")(l_input)
        
        l_conv_2 = Conv2D(32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu")(l_conv_1)
        
        l_flatten = Flatten()(l_conv_2)
        
        l_dense = Dense(256, activation="relu")(l_flatten)
        
        out_actions = Dense(self.n_actions, activation='softmax')(l_dense)
        out_value   = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()    # have to initialize before threading

        return model

    def _build_graph(self, model):
        """ Build a custom loss function for the optimization of the neural network
        
        There are three different losses that are combined, policy, value and entropy
        
        Policy looks at which actions should be chosen
        Value looks at which actions has what value in a given state
        Entropy restricts the neural network to overfit to one set of actions
        """
        
        s_t = tf.placeholder(tf.float32, shape=(None,) + self.n_state)
        a_t = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
        
        p, v = model(s_t)

        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)
        loss_value  = self.LOSS_V * tf.square(advantage)
        entropy = self.LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keepdims=True)    

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)
        
        return s_t, a_t, r_t, minimize

    def optimize(self):
        "Optimize the neural network by taking a batch of observations provided by the agents"
        
        if len(self.train_queue[0]) < self.MIN_BATCH:
            time.sleep(0)    # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < self.MIN_BATCH:    # more thread could have passed without lock
                return                                     # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.stack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.stack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*self.MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + self.GAMMA_N * v * s_mask    # set v to 0 where s_ is terminal state
        
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        "Handles the incoming observations"
        
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(self.none_state)
                self.train_queue[4].append(0.)
            else:    
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        "Get policy and value for state"
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        "Get policy for state"
        with self.default_graph.as_default():
            p, v = self.model.predict(s)        
            return p

    def predict_v(self, s):
        "Get value for state"
        with self.default_graph.as_default():
            p, v = self.model.predict(s)        
            return v
        
    def save_weights(self, name):
        "Save the current weights of the neural network"
        try:
            self.model.save_weights(name)
        except KeyboardInterrupt:
            self.model.save_weights(name) 
            raise