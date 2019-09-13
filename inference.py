import tensorflow as tf
from config import *
import tensorflow_hub as hub

class quad_model:

    # Create model
    def __init__(self, margins, marginc, marginn, embedding):
        self.anchor = tf.placeholder(tf.int32, [None,])
        self.comp = tf.placeholder(tf.int32, [None,])
        self.sim= tf.placeholder(tf.int32, [None,])
        self.neg = tf.placeholder(tf.int32, [None,])
        self.margins = margins
        self.marginc = marginc
        self.marginn = marginn
        self.embed = embedding

        with tf.variable_scope("pretrain") as scope:
            self.anchor_pre = tf.nn.embedding_lookup(self.embed, self.anchor)
            self.comp_pre = tf.nn.embedding_lookup(self.embed, self.comp)
            self.sim_pre = tf.nn.embedding_lookup(self.embed, self.sim)
            self.neg_pre = tf.nn.embedding_lookup(self.embed, self.neg)

            self.anchor_pre = tf.cast(self.anchor_pre, dtype=tf.float32)
            self.comp_pre = tf.cast(self.comp_pre, dtype=tf.float32)
            self.sim_pre = tf.cast(self.sim_pre, dtype=tf.float32)
            self.neg_pre = tf.cast(self.neg_pre, dtype=tf.float32)



        with tf.variable_scope("quad") as scope:
            self.oa = self.network(self.anchor_pre)
            scope.reuse_variables()
            self.oc = self.network(self.comp_pre)
            scope.reuse_variables()
            self.os = self.network(self.sim_pre)
            scope.reuse_variables()
            self.on = self.network(self.neg_pre)

        self.loss = self.triplet_loss()
        #self.normalized_dist()

    def network(self, x):
        fc1 = self.fc_layer(x, n_hidden_1, "fc1")     #input, output, weight
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, n_hidden_2, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, n_hidden_3, "fc3")
        # ac3 = tf.nn.relu(fc3)
        # fc4 = self.fc_layer(ac3, 1, "fc4")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        #initer = tf.truncated_normal_initializer(stddev=0.01)
        initer = tf.glorot_normal_initializer(seed=None)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.00, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc




    def triplet_loss(self):
        self.oa = tf.nn.l2_normalize(self.oa, axis=1)
        self.oc = tf.nn.l2_normalize(self.oc, axis=1)
        self.on = tf.nn.l2_normalize(self.on, axis=1)
        self.os = tf.nn.l2_normalize(self.os, axis=1)

        dist_comp = tf.pow(tf.subtract(self.oa, self.oc), 2)
        self.dist_comp_s = tf.reduce_sum(dist_comp, axis=1)
        self.dist_comp_norm = tf.sqrt(self.dist_comp_s)

        loss_comp1 = tf.nn.relu(self.dist_comp_s - self.marginc)

        loss_comp2 = tf.nn.relu(self.margins - self.dist_comp_s)

        dist_sim = tf.pow(tf.subtract(self.oa, self.os), 2)
        self.dist_sim_s = tf.reduce_sum(dist_sim, axis=1)
        self.dist_sim_norm = tf.sqrt(self.dist_sim_s)

        #loss_sim = tf.nn.relu(self.dist_sim_s - self.margins)
        loss_sim = self.dist_sim_s

        dist_neg = tf.pow(tf.subtract(self.oa, self.on), 2)
        self.dist_neg_s = tf.reduce_sum(dist_neg, axis=1)
        self.dist_neg_norm = tf.sqrt(self.dist_neg_s)
        loss_neg = tf.nn.relu(self.marginn - self.dist_neg_s)
        #loss_neg = self.dist_neg_s

        # TODO: check fo greater than margin_s also
        check1 = tf.cast(self.dist_comp_s < self.marginc, dtype=tf.int32)
        check2 = tf.cast(self.dist_comp_s > self.margins, dtype=tf.int32)
        self.y_pred_comp = tf.multiply(check1, check2)
        self.y_pred_sim = self.dist_sim_s < self.margins
        self.y_pred_neg = self.dist_neg_s > self.marginc
        self.y_pred = self.dist_comp_s < self.dist_neg_s

        self.y_ones = tf.ones_like(self.y_pred_comp)

        losses = loss_sim + loss_comp1 + loss_comp2 + loss_neg#- loss_neg#+ loss_sim + loss_comp2 - loss_neg

        loss = tf.reduce_mean(losses, name="loss")
        return loss

