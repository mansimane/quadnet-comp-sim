
""" Triplet network implementation using Tensorflow for CTL (Complete The Look) project.

    This code takes data in the form of embeddings for anchor, positive and negative items and trains a
    triplet network with this data. Train and Test data are supposed to segregated before and kept separately
    in train and test folder.
"""

__author__ = "Mansi Mane"
__email__ = "mansi.mane@walmartlabs.com"

#import system things
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.extend(['./model/'])

#import helpers
#import visualize
from config import *
from input_fn import load_datasets_triplet
import inference
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import seaborn as sns
import pandas as pd
import time

'''
Config
'''
start = time.time()
pos_acc_array = [None] * len(MARGIN_ARRAY_c)
neg_acc_array = [None] * len(MARGIN_ARRAY_c)
precision_array = [None] * len(MARGIN_ARRAY_c)
acc_array = [None] * len(MARGIN_ARRAY_c)
np.random.seed(seed)


for idx_mar, margin in enumerate(MARGIN_ARRAY_c):

    sess = tf.InteractiveSession()
    tf.set_random_seed(seed)
    model_spec = {}
    dist_comp_array = []
    dist_sim_array = []
    dist_neg_array = []
    dist_comp_array_tst = []
    dist_sim_array_tst = []
    dist_neg_array_tst = []

    if REAL_DATA == False:
        os.environ['TFHUB_CACHE_DIR'] = '/Users/m0m02d5/Desktop/ctl_demo/universal_sentence_encoder_large'
        path = '/Users/m0m02d5/Documents/data/amazon/Clothing_Shoes_and_Jewelry/'
        meta_path = '/Users/m0m02d5/Documents/data/amazon/Clothing_Shoes_and_Jewelry/meta_Clothing_Shoes_and_Jewelry.csv'
        embedding_path = '/Users/m0m02d5/Documents/data/amazon/Clothing_Shoes_and_Jewelry/text/id_embedding.csv'
        embedding_np_path = "/Users/m0m02d5/Documents/data/amazon/Clothing_Shoes_and_Jewelry/id_embedding.npy"


    # meta_df = pd.read_csv(meta_path, header=None, names=['item_id', 'cat', 'title', 'url'],
    #                       delimiter='\01', dtype= {'item_id': str, 'cat': str, 'title': str, 'url': str})
    # meta_df = meta_df.drop(['cat','url'], axis=1)
    # meta_df = meta_df.dropna()
    # a = np.array(meta_df['item_id'])
    # a = a[:, np.newaxis]
    # b = np.array(meta_df['title'])
    # b = b[:, np.newaxis]
    # c = np.concatenate((a,b), axis=1)
    itemid_title = tf.convert_to_tensor('junk')
    if not os.path.exists(embedding_np_path):

        embedding_df = pd.read_csv(embedding_path, header=None, names=['item_id', 'ev'], delimiter='\01',\
                                   dtype= {'item_id': str, 'ev': str})

        embedding_df['ev'] = embedding_df['ev'].apply(eval)
        embedding_df['ev'] = embedding_df['ev'].apply(np.array)
        def reshape(x):
            x = x[np.newaxis, :]
            return x

        embedding_df['ev'] = embedding_df['ev'].apply(reshape)
        ev = embedding_df['ev'].tolist()
        ev = np.array(ev)
        ev = np.squeeze(ev, axis=1)
        np.save(embedding_np_path, ev)
    else:
        ev = np.load(embedding_np_path)
    embedding = tf.Variable(ev, trainable=False, name='embedding')

    # setup  network
    print("Using Triplet model")
    batch_tr, batch_tst, tr_iter_init_op, tst_iter_init_op = load_datasets_triplet(path, itemid_title)
    # sess.run(tr_iter_init_op)
    # p, q, r, s = sess.run(batch_tr)
    # print(p,q,r,s)

    model = inference.quad_model(margins=MARGIN_ARRAY_s[idx_mar], marginc=MARGIN_ARRAY_c[idx_mar], marginn=MARGIN_ARRAY_n[idx_mar], embedding=embedding)

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    tr_loss_minimize = optimizer.minimize(model.loss, global_step=global_step)
    saver = tf.train.Saver()
    if not os.path.exists("summaries/"):
        os.mkdir("summaries/")
    if not os.path.exists("figures/"):
        os.mkdir("figures/")

    summary_path = "summaries/" + "margin_" + str(margin)
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)
    if not os.path.exists(os.path.join(summary_path,'train')):
        os.mkdir(os.path.join(summary_path,'train'))
    if not os.path.exists(os.path.join(summary_path,'test')):
        os.mkdir(os.path.join(summary_path,'test'))
    if REAL_DATA == False:
        prev_summaries = os.listdir(os.path.join(summary_path,'test'))
        if len(prev_summaries) > 1:
            old_summary_path = summary_path + '_old'
            if not os.path.exists(old_summary_path):
                os.mkdir(old_summary_path)
                os.mkdir(os.path.join(old_summary_path, 'train'))
                os.mkdir(os.path.join(old_summary_path, 'test'))

            for summary in prev_summaries:
                os.rename(os.path.join(summary_path,'test', summary), os.path.join(old_summary_path,'test', summary))
            prev_summaries = os.listdir(os.path.join(summary_path, 'train'))
            for summary in prev_summaries:
                os.rename(os.path.join(summary_path,'train', summary), os.path.join(old_summary_path,'train', summary))
            print('Moved previous summary runs to ', old_summary_path)

    tr_summ_writer = tf.summary.FileWriter(os.path.join(summary_path, 'train'), sess.graph)
    tst_summ_writer = tf.summary.FileWriter(os.path.join(summary_path, 'test'), sess.graph)
    model_spec['model'] = model

    '''
    Creting tensorboard summaries
    '''
    # Name scope allows you to group various summaries together
    # Summaries having the same name_scope will be displayed on the same row
    with tf.name_scope('performance'):
        # Summaries need to be displayed
        # Whenever you need to record the loss, feed the mean loss to this placeholder

        # Create a scalar summary object for the loss so it can be displayed
        loss_summary = tf.summary.scalar('loss', model.loss)

        # Putting y_true_comp everywhere as we just want to put tf.ones everyone, but can't hardcode shape to batchsize
        acc_comp_ph, acc_comp_op = tf.metrics.accuracy(labels=model.y_ones, predictions=model.y_pred_comp)
        acc_sim_ph, acc_sim_op = tf.metrics.accuracy(labels=model.y_ones, predictions=model.y_pred_sim)
        acc_neg_ph, acc_neg_op = tf.metrics.accuracy(labels=model.y_ones, predictions=model.y_pred_neg)
        acc_ph, acc_op = tf.metrics.accuracy(labels=model.y_ones, predictions=model.y_pred)

        acc_summary_comp = tf.summary.scalar('comp_accuracy', acc_comp_ph)
        acc_summary_sim = tf.summary.scalar('sim_accuracy', acc_sim_ph)
        acc_summary_neg = tf.summary.scalar('neg_accuracy', acc_neg_ph)
        acc_summary = tf.summary.scalar('Complementary-negative Accuracy', acc_ph)

        pr_summary = tf.summary.scalar('precision', acc_comp_ph/(acc_comp_ph + (1.0 - acc_comp_ph)))
        #pr_summary = tf.summary.scalar('precision', precision)
        recall_summary = tf.summary.scalar('recall', acc_comp_ph/(acc_comp_ph + (1.0 - acc_comp_ph)))


        # Gradient norm summary
        for g,v in grads_and_vars:
            if 'hidden' in v.name and 'weights' in v.name:
                with tf.name_scope('gradients'):
                    last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                    gradnorm_summary = tf.summary.scalar('grad_norm', last_grad_norm)
                    break

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="performance")
    metrics_init_op = tf.variables_initializer(metric_variables)

    print('Initialized metric variables')
    # Merge all summaries together
    loss_summaries =tf.summary.merge([loss_summary])
    performance_summaries = tf.summary.merge([acc_summary_comp, acc_summary_sim, acc_summary_neg, acc_summary,\
                                                 pr_summary, recall_summary])

    tf.global_variables_initializer().run()
    sess.run([tf.local_variables_initializer(), tf.tables_initializer()])

    print("Initialization took {0:.1f} sec".format(time.time() - start))

    # start training
    if not TRAIN_FROM_SCRATCH:
        saver.restore(sess, './model/triplet_v1_model_mar_0.6_epoch_-9')

    for e in range(epochs):
        start = time.time()

        sess.run(tr_iter_init_op)
        sess.run(tst_iter_init_op)

        pos_equal = 0
        neg_equal = 0
        ps_equal = 0
        while True:
            try:
                anchor_batch, comp_batch, sim_batch, neg_batch = sess.run(batch_tr)
                # When only example is remaining, it's scalar. Model expects a 1D array
                # if len(neg_batch.shape) == 0:
                #     anchor_batch = np.array([anchor_batch])
                #     comp_batch = np.array([comp_batch])
                #     sim_batch = np.array([sim_batch])
                #     neg_batch = np.array([neg_batch])

                _, tr_loss_ph, loss_summ, global_step_val, dist_comp_s, dist_sim_s, dist_neg_s, acc_comp, acc_sim, acc_neg, acc = sess.run(
                    [tr_loss_minimize, model.loss, loss_summaries, global_step,\
                     model.dist_comp_norm, model.dist_sim_norm, model.dist_neg_norm, acc_comp_op, acc_sim_op, acc_neg_op, acc_op], feed_dict={
                        model.anchor: anchor_batch,
                        model.comp: comp_batch,
                        model.sim: sim_batch,
                        model.neg: neg_batch})

                if e == (epochs-1):
                    for d in dist_comp_s:
                        dist_comp_array.append(d)
                    for d in dist_sim_s:
                        dist_sim_array.append(d)
                    for d in dist_neg_s:
                        dist_neg_array.append(d)

                if global_step_val % save_summary_steps ==0:
                    tr_summ_writer.add_summary(loss_summ, global_step_val)
                    tr_summ_writer.flush()
                    print('global_step_val ', global_step_val)
            except tf.errors.OutOfRangeError:
                break

        print("One epoch took {0:.1f} sec".format(time.time() - start))

        acc, tr_acc_comp, tr_acc_sim, tr_acc_neg,summary = sess.run([acc_ph, acc_comp_ph, acc_sim_ph, acc_neg_ph, performance_summaries])
        tr_summ_writer.add_summary(summary, e)
        tr_summ_writer.flush()

        if np.isnan(tr_loss_ph):
            print('Model diverged with loss = NaN')
            quit()

        # '''
        # Evaluate on Validation Data
        # '''
        sess.run(metrics_init_op)
        num_batch = 0
        while True:
            try:
                anchor_batch, comp_batch, sim_batch, neg_batch = sess.run(batch_tst)
                # When only example is remaining, it's scalar. Model expects a 1D array
                # if np.isscalar(neg_batch):
                #     anchor_batch = np.array([anchor_batch])
                #     comp_batch = np.array([comp_batch])
                #     sim_batch = np.array([sim_batch])
                #     neg_batch = np.array([neg_batch])

                if num_batch==0:
                    acc, acc_comp, acc_neg, acc_sim, tst_loss_ph, loss_summ, \
                    dist_comp_s, dist_sim_s, dist_neg_s = sess.run(
                        [acc_op, acc_comp_op, acc_neg_op, acc_sim_op, model.loss, loss_summaries,\
                         model.dist_comp_norm, model.dist_sim_norm, model.dist_neg_norm], feed_dict={
                            model.anchor: anchor_batch,
                            model.comp: comp_batch,
                            model.sim: sim_batch,
                            model.neg: neg_batch})
                else:
                    acc, acc_comp, acc_neg, acc_sim, tst_loss_ph, \
                    dist_comp_s, dist_sim_s, dist_neg_s = sess.run(
                        [acc_op, acc_comp_op, acc_neg_op, acc_sim_op, model.loss,\
                         model.dist_comp_norm, model.dist_sim_norm, model.dist_neg_norm], feed_dict={
                            model.anchor: anchor_batch,
                            model.comp: comp_batch,
                            model.sim: sim_batch,
                            model.neg: neg_batch})

                if e == (epochs-1):
                    for d in dist_comp_s:
                        dist_comp_array_tst.append(d)
                    for d in dist_sim_s:
                        dist_sim_array_tst.append(d)
                    for d in dist_neg_s:
                        dist_neg_array_tst.append(d)
                num_batch += 1


            except tf.errors.OutOfRangeError:
                break

        acc, acc_comp, acc_sim, acc_neg, summary = sess.run([acc_ph, acc_comp_ph, acc_sim_ph, acc_neg_ph, performance_summaries])

        tst_summ_writer.add_summary(loss_summ, global_step_val)
        tst_summ_writer.add_summary(summary, e)
        tst_summ_writer.flush()
        print ('epoch %d: train_loss %.6f, Accuracy %.6f:, tr_acc_comp: %.5f, tr_acc_sim: %.5f, tr_acc_neg: %.5f' %\
               (e, tr_loss_ph, acc, tr_acc_comp,tr_acc_sim, tr_acc_neg ))
        print('epoch %d: Test_loss %.6f, Accuracy_test %.6f, tst_acc_comp %.6f, tst_acc_sim %.6f, tst_acc_neg %.3f ' %\
              (e, tst_loss_ph, acc, acc_comp, acc_sim, acc_neg))

        print("margin: ", margin)
        sess.run(metrics_init_op)

        if REAL_DATA == True:
            if (e %10 == 0) and (e!=0) :
                sess_str = './model/triplet_v1_model_' + 'mar_' + str(margin) + '_epoch_'
                saver.save(sess, sess_str, global_step=e+1)

    # Store below metrics once all epochs for on margin are run
    precision = tr_acc_comp / (tr_acc_comp + (1.0 - tr_acc_neg))

    # Plot distance distributions
    # if len(dist_comp_array) == 1:
    print("Train: comp:", dist_comp_array)
    print("sim ", dist_sim_array)
    print("neg ", dist_neg_array)

    print("Test: comp:", dist_comp_array_tst)
    print("sim ", dist_sim_array_tst)
    print("neg ", dist_neg_array_tst)
    
    sess_str = './model/triplet_v1_model_' + 'mar_' + str(margin) + '_epoch_'
    saver.save(sess, sess_str, global_step= e + 1)

    ax = sns.distplot(dist_comp_array, hist=False, kde=True, label='Complementary')
    fig = ax.get_figure()
    sns.distplot(dist_neg_array, hist=False, kde=True, label='Negtive')
    sns.distplot(dist_sim_array, hist=False, kde=True, label='Similar')
    name = "./figures/dist_distribution_init_margin_" + str(margin) + ".pdf"
    fig.savefig(name)
    fig.clf()

    ax = sns.distplot(dist_comp_array_tst, hist=False, kde=True, label='Complementary')
    fig = ax.get_figure()
    sns.distplot(dist_neg_array_tst, hist=False, kde=True, label='Negtive')
    sns.distplot(dist_sim_array_tst, hist=False, kde=True, label='Similar')
    name = "./figures/tst_dist_distribution_init_margin_" + str(margin) + ".pdf"
    fig.savefig(name)
    fig.clf()

    pos_acc_array[idx_mar] = tr_acc_comp
    neg_acc_array[idx_mar] = tr_acc_neg
    precision_array[idx_mar] = precision
    acc_array[idx_mar] = acc
    tf.reset_default_graph()
    sess.close()

