from contextlib import contextmanager
import io, time
import tensorflow as tf

@contextmanager
def log_time():
    start = time.time()
    buff = io.StringIO()
    yield buff
    end = time.time()
    buff.seek(0)
    print("in %.3f sec. "%(end - start) + buff.read())

def emb_from_id(ids, emb, shape):
    #ids1 = tf.reshape(ids, [-1, 1])%100000
    #emb1 = tf.nn.embedding_lookup(emb, ids1)
    #print 'emb1.shape:',emb1.shape
    #emb2 = tf.reshape(emb1, shape)
    emb2 = tf.nn.embedding_lookup(emb, ids%100000)
    print 'emb2.shape:',emb2.shape
    return emb2

def decode_csv(src, dv, l):
    return tf.decode_csv(src, record_defaults=[[dv]]*l)

def tensor_from_file(Tvalue, value_type, value_length, output_shape):
    Dvalue = decode_csv(Tvalue, value_type, value_length)
    value_stack = tf.stack(Dvalue, axis=0)
    print 'value_stack.shape:',value_stack.shape
    Rvalue = tf.reshape(value_stack, output_shape)
    return Rvalue

if __name__ == '__main__':
    with log_time() as log:
        time.sleep(1)
        log.write('hellow')
