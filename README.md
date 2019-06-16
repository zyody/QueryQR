# QueryQR

1. Download datasets from https://pan.baidu.com/s/1xrH1Q_O4zGxMNUbwNITpyg. The data is stored as the tfrecord format, with feature columns defined as:
```bash
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
}
```
where _click_ is the label, behavior features in the user feature field are extracted from _cate_list, catelevel1_list, item_list, item_features_list, time_list, action_list, item_name_hash_list, i2q_term_hash_list_. Other features in the user, item, context feature fields are extracted from the remaining feature columns.  
  
  
  
2. Suppose the download path is: ~/ijcai592_dataset, then run: 
```bash
python train_tfrecord.py --buckets "~/ijcai592_dataset/" --checkpointDir "~/log/"
```
