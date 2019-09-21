import tensorflow as tf
from config import *
import numpy as np
import  os

def _parse_function_triplet(line, itemid_title):
    # i: anchor items
    # c: complementary item
    # s: similar item
    # n: negative item
    record_defaults = [[""]]*5
    _,i, c, s, n = tf.decode_csv(line,
                                record_defaults,
                                field_delim='\01',
                                use_quote_delim=False)

    i = tf.string_to_number(i, tf.int32)
    c = tf.string_to_number(c, tf.int32)
    s = tf.string_to_number(s, tf.int32)
    n = tf.string_to_number(n, tf.int32)
    # graph = tf.get_default_graph()
    # itemid_title = graph.get_tensor_by_name("itemid_title")
    # i_idx = tf.where(tf.equal(itemid_title[:,0], i))
    # c_idx = tf.where(tf.equal(itemid_title[:,0], c))
    # s_idx = tf.where(tf.equal(itemid_title[:,0], s))
    # n_idx = tf.where(tf.equal(itemid_title[:,0], n))
    # 
    # i_title = itemid_title[i_idx[0][0], 1]
    # c_title = itemid_title[c_idx[0][0], 1]
    # s_title = itemid_title[s_idx[0][0], 1]
    # n_title = itemid_title[n_idx[0][0], 1]

    # return i_title, c_title, s_title, n_title
    return i, c, s, n


def load_datasets_triplet(path, itemid_title):

    train_file_names = os.listdir(path + 'train/')
    train_file_names = [path + 'train/' + file_name for file_name in train_file_names]

    dataset_tr = tf.data.Dataset.from_tensor_slices(train_file_names).shuffle(buffer_size=10)
    dataset_tr = dataset_tr.flat_map(lambda filename: tf.data.TextLineDataset(filename))

    dataset_tr = dataset_tr.map(lambda x: _parse_function_triplet(x, itemid_title), num_parallel_calls=1)

    dataset_tr = dataset_tr.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=RANDOM_SEED)
    dataset_tr = dataset_tr.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
    dataset_tr = dataset_tr.prefetch(buffer_size=BATCH_SIZE * PREFETECH_BUFFER_SIZE)

    iter_tr = dataset_tr.make_initializable_iterator()
    batch_tr = iter_tr.get_next()

    tst_file_names = os.listdir(path + 'test/')
    idx = min(5, len(tst_file_names))

    print('5 Test files : ', tst_file_names[0:idx])
    tst_file_names = [path + 'test/' + file_name for file_name in tst_file_names]

    dataset_tst = tf.data.TextLineDataset(tst_file_names[0])
    dataset_tst = dataset_tst.map(lambda x: _parse_function_triplet(x, itemid_title), num_parallel_calls=1)
    dataset_tst = dataset_tst.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
    dataset_tst = dataset_tst.prefetch(buffer_size=BATCH_SIZE * PREFETECH_BUFFER_SIZE)

    iter_tst = dataset_tst.make_initializable_iterator()
    batch_tst = iter_tst.get_next()

    tr_iter_init_op = iter_tr.initializer
    tst_iter_init_op = iter_tst.initializer
    return batch_tr, batch_tst, tr_iter_init_op, tst_iter_init_op


def parse_function_elmo_embedding(line):
    record_defaults = [[""]] * 3074
    line_parse = tf.decode_csv(line, record_defaults, field_delim='\01')
    embedding = tf.stack(line_parse[1:-1])
    embedding = tf.string_to_number(embedding, tf.float32)
    item_id = line_parse[-1]
    return item_id, embedding

def parse_function_elmo_embedding_top_few(line):
    record_defaults = [[""]] * 10
    line_parse = tf.decode_csv(line, record_defaults, field_delim='\01',\
                               use_quote_delim=False)
    catalog_item_id, super_dept, dept, cat, subcat, title, product_type, room, primary_price, ev = line_parse
    ev = tf.strings.regex_replace(ev, '\[', '')
    ev = tf.strings.regex_replace(ev, '\]', '')
    #embedding = tf.stack(line_parse[3])
    ev = tf.string_split(source=[ev], delimiter=',').values
    ev = tf.string_to_number(ev, tf.float32)

    return catalog_item_id, super_dept, dept, cat, subcat, title, product_type, room, primary_price, ev

def load_elmo_item_embeddings(path):
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(parse_function_elmo_embedding_top_few)
    dataset = dataset.batch(BATCH_SIZE)

    iter = dataset.make_initializable_iterator()
    batch_data = iter.get_next()

    iter_init_op = iter.initializer

    return batch_data, iter_init_op


def load_guse_item_embeddings(path):
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(parse_function_guse_embedding_top_few)
    dataset = dataset.batch(BATCH_SIZE)

    iter = dataset.make_initializable_iterator()
    batch_data = iter.get_next()

    iter_init_op = iter.initializer

    return batch_data, iter_init_op