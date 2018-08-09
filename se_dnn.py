#!/usr/bin/python
##-*- coding:UTF-8 -*-
import tensorflow as tf
import scipy.io as scio
import os

MODE = 'test'

def process_file_list(file_list):
    fid = open(file_list, 'r')
    proc_file_list = []
    lines = fid.readlines()
    for line in lines:
        proc_file_list.append(line.rstrip('\n'))
    return proc_file_list

def read_and_decode_train(filename, input_dim, label_dim, num_epochs):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    _, features = tf.parse_single_sequence_example(
        serialized_example,
        sequence_features={
           'inputs': tf.FixedLenSequenceFeature([input_dim], tf.float32),
           'labels': tf.FixedLenSequenceFeature([label_dim], tf.float32)
           }
        )
    feats = features['inputs']
    labels = features['labels']
    return feats, labels

def read_and_decode_test(filename, input_dim, num_epochs):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    _, features = tf.parse_single_sequence_example(
        serialized_example,
        sequence_features={
            'inputs': tf.FixedLenSequenceFeature([input_dim], tf.float32)
            }
        )
    feats = features['inputs']
    return feats

def splice_feats(feats, l, r):
    sfeats = []
    row = tf.shape(feats)[0]
    for i in range(l, 0, -1):
        f1 = tf.slice(feats, [0, 0], [row - i, -1])
        for j in range(i):
            f1 = tf.pad(f1, [[1, 0], [0, 0]], mode='SYMMETRIC')
        sfeats.append(f1)

    sfeats.append(feats)
    for i in range(1, r + 1):
        f1 = tf.slice(feats, [i, 0], [-1, -1])
        for j in range(i):
            f1 = tf.pad(f1, [[0, 1], [0, 0]], mode='SYMMETRIC')
        sfeats.append(f1)
    return tf.concat(sfeats, 1)

def get_mini_batch(sess, coord, file_list, input_dim=257, label_dim=257, l=0, r=0, batch_size=1, num_threads=4, num_epochs=1):
    n_input = (l + 1 + r) * input_dim
    n_output = label_dim

    filename = process_file_list(file_list)
    feats, labels = read_and_decode_train(filename, input_dim, label_dim, num_epochs)
    
    sess.run(tf.local_variables_initializer())
    sfeats = splice_feats(feats, l, r)

    slice_queue = tf.RandomShuffleQueue(
        capacity=batch_size * 50,
        min_after_dequeue=0,
        dtypes=['float', 'float'],
        shapes=[[n_input, ], [n_output, ]]
    )

    batch_x, batch_y = slice_queue.dequeue_many(batch_size)#出列   
    enqueue = [slice_queue.enqueue_many([sfeats, labels])] * num_threads#入列

    #创建一个队列管理器QueueRunner，向slice_queue中添加元素。目前使用num_threads=4个线程:
    qr = tf.train.QueueRunner(slice_queue, enqueue)
    qr.create_threads(sess, coord=coord, start=True)

    return batch_x, batch_y

def initialize_parameters(): 
    tf.set_random_seed(1) 

    W1 = tf.get_variable('W1', [1285,2048], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable('b1', [2048], initializer = tf.zeros_initializer())
    W2 = tf.get_variable('W2', [2048,2048], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable('b2', [2048], initializer = tf.zeros_initializer())
    W3 = tf.get_variable('W3', [2048,2048], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable('b3', [2048], initializer = tf.zeros_initializer())
    W4 = tf.get_variable('W4', [2048,2048], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable('b4', [2048], initializer = tf.zeros_initializer())
    W5 = tf.get_variable('W5', [2048,257], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b5 = tf.get_variable('b5', [257], initializer = tf.zeros_initializer())

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2,
                  'W3': W3,
                  'b3': b3,
                  'W4': W4,
                  'b4': b4,
                  'W5': W5,
                  'b5': b5}

    return parameters

def forward_propagation(x, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
  
    f1 = tf.nn.relu(tf.matmul(x,W1)+b1)
    #f1 = tf.nn.dropout(f1, 0.7)
    f2 = tf.nn.relu(tf.matmul(f1,W2)+b2)
    #f2 = tf.nn.dropout(f2, 0.7)
    f3 = tf.nn.relu(tf.matmul(f2,W3)+b3)
    #f3 = tf.nn.dropout(f3, 0.7)
    f4 = tf.nn.relu(tf.matmul(f3,W4)+b4)
    #f4 = tf.nn.dropout(f4, 0.7)
    y = tf.matmul(f4,W5)+b5                                          

    return y

def train():

    file_list = 'train_tf.lst'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()

    sess.run(tf.local_variables_initializer())
    sfeats, labels = get_mini_batch(sess, coord, file_list, l=2, r=2, batch_size=512, num_epochs=35)
    parameters = initialize_parameters()
    output = forward_propagation(sfeats, parameters)
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    global_step = tf.Variable(0,trainable=False)
    # batch_size=512
    # loss = tf.reduce_sum(tf.pow(tf.subtract(output,labels),2.0))/batch_size
    loss = tf.reduce_mean(tf.square(labels - output))
    tf.summary.scalar('loss',loss)
    
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    #learning_rate = 0.005  
    learning_rate = tf.train.exponential_decay(
        0.005,
        global_step,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    summary_writer = tf.summary.FileWriter('./log', sess.graph)
    sess.run(tf.local_variables_initializer())   
    sess.run(tf.global_variables_initializer())
    
    mean_loss = 0.0
    try:
        while not coord.should_stop():
            summary, _, _loss, step = sess.run([merged, train_step, loss, global_step])
            summary_writer.add_summary(summary,step) 
            mean_loss+=_loss
            if step != 0 and step % 100 == 0:
                print("step: %d , loss: %g"%(step, _loss))
                mean_loss = mean_loss/100
                print('mean_loss:',mean_loss)
                mean_loss = 0.0
            if step != 0 and step % 10000 == 0:
                save_path = "./model/se_model"
                saver.save(sess, save_path, step)
                print("step: %d , the model saved in path: %s" %(step, save_path))
    except tf.errors.OutOfRangeError:
        print('train done')
        return
    finally:
        coord.request_stop()

    coord.join(thread)
    sess.close()

def test():
    list_path = 'test_tf.lst'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config = config)
    coord = tf.train.Coordinator()
    file_list = process_file_list(list_path)
    feats = read_and_decode_test(file_list,257,1)
    sfeats = splice_feats(feats,2,2)
    sess.run(tf.local_variables_initializer())   
    sess.run(tf.global_variables_initializer())
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    parameters = initialize_parameters()
    count = 0
    try:
        while not coord.should_stop():
            x = sess.run(sfeats)
            y = forward_propagation(x, parameters)
            saver = tf.train.Saver()
            saver.restore(sess,'./model/se_model-380000')
            labels = sess.run(y)
            print(labels)
            print(labels.shape)
            file_path = file_list[count]
            filename = os.path.basename(file_path)
            (tfname, _) = os.path.splitext(filename)
            print(tfname)
            (name, _) = os.path.splitext(tfname)
            print(name)
            scio.savemat('./mat_result/'+name+'.mat',{name:labels})
            print('count:',count)
            count += 1
    except tf.errors.OutOfRangeError:
        print('test done')
        return
    finally:
        coord.request_stop()

    coord.join(thread)
    sess.close()

if __name__ == "__main__":
    if MODE == 'train':
        train()
    if MODE == 'test':
        test()