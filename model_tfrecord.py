import os

import numpy as np
import tensorflow as tf
#from trnn import TGRUCell

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Seq2Point:
    def __init__(self,table_name="", checkpoint_dir="",
                 epochs=100, batch_size=500, seq_length=100, feature_dim=56, term_length = 15, item_features_dim = 34,
                 query_features_dim=7, m1=1, m2=1, m3=1, is_train=True, is_debug = True):

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.001

        # IO config
        self.table_name = table_name
        self.checkpointDir = checkpoint_dir
        # param config
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.query_features_dim = query_features_dim
        self.item_features_dim = item_features_dim
        self.output_length = 1
        self.feature_dim = feature_dim
        self.term_length = term_length
        self.is_train = is_train
        self.is_debug = is_debug

        # model_param
        self.rnn_cell_dim = 256
        self.dense_dim = 50#128
        self.attention_dim = 10

        self.short_dim = 4

        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.X = []
        self.X_last = ''
        self.brand_ids = ''
        self.fc1 = ''
        self.fc2 = ''
        self.embed_tmp1 = ''
        self.embed_tmp2 = ''
        self.X1 = ''
        self.X2 = ''

    # Init Graph
    def build_input(self):
        self.Ruserid, self.Rcate_list, self.Rcatelevel1_list, self.Rbrand_list, self.Ritem_list, self.Rtime_list, self.Raction_list, self.Ritem_name_hash_list, self.Ri2q_term_hash_list, self.Ritem_features_list, self.Rquery_hash_list, self.Rquery_features, self.Rqid, self.Rcateid1, self.Rcateid2, self.Rcateid3, self.Rscore_features, self.Rxftrl_features, self.Rmatchtype, self.Rcateid, self.Rtriggerid, self.Rage_class, self.Rbaby_stage, self.Rcareer_type, self.label_value, self.label_oh = self.__data__()

    def build_graph(self):
        self.pred_logit = self.__build_graph__()
        self.loss, self.optimizer = self.__loss_optimizer__()

    def build_summary(self, name='train'):
        pset, mset = self.__evaluation__()
        self.positive_score, self.pred_binary = pset
        self.acc, self.precision, self.recall, self.auc = mset
        self.summary_op = self.__add_summary__(name)

    def __add_summary__(self, name):
        print('summary')
        summary = [
            tf.summary.scalar(name + '/loss', self.loss),
            tf.summary.scalar(name + '/metrics/acc', self.acc),
            tf.summary.scalar(name + '/metrics/precision', self.precision),
            tf.summary.scalar(name + '/metrics/recall', self.recall),
            tf.summary.scalar(name + '/metrics/auc', self.auc),
        ]
        summary_op = tf.summary.merge(summary)
        return summary_op

    def __evaluation__(self):
        print('evaluation')
        positive_score = tf.slice(self.pred_logit, [0, 1], [-1, 1])
        pred_binary = tf.cast(tf.round(positive_score), tf.int32)

        _, acc = tf.metrics.accuracy(self.label_value, pred_binary)
        _, precision = tf.metrics.precision(self.label_value, pred_binary)
        _, recall = tf.metrics.recall(self.label_value, pred_binary)
        _, auc = tf.metrics.auc(self.label_value, positive_score)
        return (positive_score, pred_binary), (acc, precision, recall, auc)

    def __train__(self, loss, optimizer=None):
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(loss, global_step=self.global_step)

    def __loss_optimizer__(self):
        print('make loss')
        loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label_oh, logits=self.pred_logit)
        if self.is_train:
            optimizer = self.__train__(loss)
            return loss, optimizer
        else:
            return loss, None

    def __build_graph__(self):
        print('Build Graph')

        with tf.variable_scope('embedding', initializer=tf.zeros_initializer()):
            self.embedding_catelevel1id = tf.get_variable('catelevel1_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_termid = tf.get_variable('term_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())

            catelevel1_emb = tf.nn.embedding_lookup(self.embedding_catelevel1id, self.Rcatelevel1_list%100000)
            item_name_hash_emb = tf.nn.embedding_lookup(self.embedding_termid, self.Ritem_name_hash_list%100000)
            i2q_term_hash_emb = tf.nn.embedding_lookup(self.embedding_termid, self.Ri2q_term_hash_list%100000)
            query_hash_emb = tf.nn.embedding_lookup(self.embedding_termid, self.Rquery_hash_list%100000)

            cate1_emb = tf.nn.embedding_lookup(self.embedding_catelevel1id, self.Rcateid1%100000)
            cate1_emb = tf.reshape(cate1_emb, [self.batch_size, self.feature_dim])

            cate2_emb = tf.nn.embedding_lookup(self.embedding_catelevel1id, self.Rcateid2%100000)
            cate2_emb = tf.reshape(cate2_emb, [self.batch_size, self.feature_dim])

            cate3_emb = tf.nn.embedding_lookup(self.embedding_catelevel1id, self.Rcateid3%100000)
            cate3_emb = tf.reshape(cate3_emb, [self.batch_size, self.feature_dim])

            X = tf.concat([catelevel1_emb, tf.reduce_mean(item_name_hash_emb, 2), tf.reduce_mean(i2q_term_hash_emb, 2)], axis=2)
            self.X_last = tf.concat([(cate1_emb + cate2_emb + cate3_emb)/3, tf.reduce_mean(query_hash_emb, 1)], axis=1)

            self.X = tf.unstack(X, axis=1)
            self.Xone = self.X[0]
            self.X1 = self.X[-1]
            self.X2 = self.X[-2]

            #####       Encoder Level       #####
            
            fw_cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_cell_dim)
            bw_cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_cell_dim)

            bi_output, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, self.X,
                                                                                    dtype=tf.float32)
            #####       Attention Level       #####
            # 19*cell
            outputs = tf.stack(bi_output)  # L, N, D
            state = tf.concat([state_fw, state_bw, self.X_last], 1)  # N, 3*D
            with tf.variable_scope("attention", initializer=tf.random_normal_initializer()):
                att_w = tf.get_variable('att_w', [self.rnn_cell_dim * 2, self.attention_dim], tf.float32)
                att_u = tf.get_variable('att_u', [self.rnn_cell_dim * 2 + self.feature_dim * 2, self.attention_dim],
                                        tf.float32)
                att_b = tf.get_variable('att_b', [self.attention_dim], tf.float32)
                att_v = tf.get_variable('att_v', [self.attention_dim, 1], tf.float32)

                att_ht = tf.tensordot(outputs, att_w, axes=1)  # L, N , 10
                att_h = tf.tensordot(state, att_u, axes=1)  # N, 10
                e = att_ht + att_h + att_b  # L, N, 10
                e = tf.transpose(e, perm=[1, 0, 2])  # N,L, 10
                e = tf.nn.elu(e)
                e = tf.tensordot(e, att_v, axes=[[2], [0]])
                e = tf.reshape(e, shape=[self.batch_size, self.seq_length])  # N, L
                att_value = tf.nn.softmax(e)

                weighted_ht = tf.transpose(outputs, perm=[2, 1, 0]) * att_value
                att_outputs = tf.transpose(tf.reduce_sum(weighted_ht, axis=2), perm=[1, 0])

            self.final_out = tf.concat([state_fw, state_bw, att_outputs, self.X_last], 1)  # N, 3*D

            #####       Dense Classification    #####
            self.fc1 = tf.layers.dense(self.final_out, self.dense_dim, activation=tf.nn.relu, name='fc1', kernel_initializer = tf.random_normal_initializer())
            self.fc2 = tf.layers.dense(self.fc1, 2, activation=None, name='fc2')
            pred_logit = tf.nn.softmax(self.fc2)
            return pred_logit

    def __load_data__(self):
        # pos,neg sample combine     by muming
        if self.is_train:
            table = os.path.join(self.table_name, 'train/*.tf.*')
        else:
            table = os.path.join(self.table_name, 'test/*.tf.*')
        print("reading data from", table)

        selected_cols = 'user_id,shop_list,cate_list,catelevel1_list,brand_list,item_list,item_features_list,time_list,action_list,item_name_hash_list,i2q_term_hash_list,pv_reach_time,click,qid,query_features,query_hash_list,l1_cateid1,l1_cateid2,l1_cateid3,matchtype,cateid,triggerid,score_features,age_class,baby_stage,career_type,xftrl_features'
 
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(table),
														num_epochs=self.epochs, shuffle=True)
        reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
										   features={
											   'user_id': tf.FixedLenFeature([1], tf.int64),
											   'shop_list': tf.FixedLenFeature([100], tf.int64),
											   'cate_list': tf.FixedLenFeature([100], tf.int64),
											   'catelevel1_list': tf.FixedLenFeature([100], tf.int64),
											   'brand_list': tf.FixedLenFeature([100], tf.int64),
											   'item_list': tf.FixedLenFeature([100], tf.int64),
											   'item_features_list': tf.FixedLenFeature([34*100], tf.float32),
											   'time_list': tf.FixedLenFeature([100], tf.int64),
											   'action_list': tf.FixedLenFeature([100], tf.int64),
											   'item_name_hash_list': tf.FixedLenFeature([15*100], tf.int64),
											   'i2q_term_hash_list': tf.FixedLenFeature([15*100], tf.int64),
											   'pv_reach_time': tf.FixedLenFeature([1], tf.int64),
											   'click': tf.FixedLenFeature([1], tf.float32),
											   'qid': tf.FixedLenFeature([1], tf.int64),
											   'query_features': tf.FixedLenFeature([7], tf.float32),
											   'query_hash_list': tf.FixedLenFeature([15], tf.int64),
											   'l1_cateid1': tf.FixedLenFeature([1], tf.int64),
											   'l1_cateid2': tf.FixedLenFeature([1], tf.int64),
											   'l1_cateid3': tf.FixedLenFeature([1], tf.int64),
											   'matchtype': tf.FixedLenFeature([1], tf.int64),
											   'cateid': tf.FixedLenFeature([1], tf.int64),
											   'triggerid': tf.FixedLenFeature([1], tf.int64),
											   'score_features': tf.FixedLenFeature([2], tf.float32),
											   'age_class': tf.FixedLenFeature([1], tf.int64),
											   'baby_stage': tf.FixedLenFeature([1], tf.int64),
											   'career_type': tf.FixedLenFeature([1], tf.int64),
											   'xftrl_features': tf.FixedLenFeature([14], tf.float32)
										   })
        var_list = [features[x] for x in selected_cols.split(',')]
        
        capacity = 20000 + 3 * self.batch_size
        tmp = tf.train.shuffle_batch(var_list,
                                     batch_size=self.batch_size,
                                     capacity=capacity,
                                     min_after_dequeue=20000,
                                     num_threads=8)

        Tuser_id, Tshop_list, Tcate_list, Tcatelevel1_list, Tbrand_list, Titem_list, Titem_features_list, Ttime_list, Taction_list, Titem_name_hash_list, Ti2q_term_hash_list, Tpv_reach_time, Tclick, Tqid, Tquery_features, Tquery_hash_list, Tl1_cateid1, Tl1_cateid2, Tl1_cateid3, Tmatchtype, Tcateid, Ttriggerid, Tscore_features, Tage_class, Tbaby_stage, Tcareer_type, Txftrl_features = tmp
        
        Ritem_name_hash_list = tf.reshape(Titem_name_hash_list, [self.batch_size, self.seq_length, self.term_length])           
        Ri2q_term_hash_list = tf.reshape(Ti2q_term_hash_list, [self.batch_size, self.seq_length, self.term_length])
        Ritem_features_list = tf.reshape(Titem_features_list, [self.batch_size, self.seq_length, self.item_features_dim])

        label_value = tf.reshape(Tclick, shape=[-1])
        label_oh = tf.one_hot(indices=tf.cast(label_value, tf.int32), depth=2, dtype=tf.float32)

        print 'Tcate_list.shape():',Tcate_list.shape

        return Tuser_id, Tcate_list, Tcatelevel1_list, Tbrand_list, Titem_list, Ttime_list, Taction_list, Ritem_name_hash_list, Ri2q_term_hash_list, Ritem_features_list, Tquery_hash_list, Tquery_features, Tqid, Tl1_cateid1, Tl1_cateid2, Tl1_cateid3, Tscore_features, Txftrl_features, Tmatchtype, Tcateid, Ttriggerid, Tage_class, Tbaby_stage, Tcareer_type, label_value, label_oh

    def __data__(self):
        print('load data')
        return self.__load_data__()
