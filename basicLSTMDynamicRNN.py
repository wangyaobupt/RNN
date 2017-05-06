import numpy as np
import tensorflow as tf
import os

def removeFileInDir(targetDir): 
    for file in os.listdir(targetDir): 
        targetFile = os.path.join(targetDir,  file) 
        if os.path.isfile(targetFile):
            print ('Delete Old Log FIle:', targetFile)
            os.remove(targetFile)
        elif os.path.isdir(targetFile):
            print ('Delete olds in log dir: ', targetFile)
            removeFileInDir(targetFile)

def genTrainAndTestData(numTrainData, numTestData):
    train_data,train_label = genUTData(numTrainData)
    test_data,test_label = genUTData(numTestData)
    return train_data, train_label,test_data,test_label

#UT data generator: each item is a sequence with length = 10, such data can be classifies into 2 classes
def genUTData(num):
    data = np.zeros((num, 10, 1))
    label = np.zeros((num, 2))
    for index in range(0, num):
        if index % 2 == 0:
            label[index][0] = 1
            for i in range(0, 5):
                data[index][i][0] = i
            for i in range(5,10):
                data[index][i][0] =  np.random.randn(1)
        else:
            label[index][1] = 1
            data[index] = np.random.randn(10, 1)

    return data, label

if __name__ == '__main__':
    # shape of input and output
    batch_size = 100
    n_train_size = 10*batch_size
    n_test_size  = batch_size
    n_classes = 2
    length_of_input_sequence = 10

    # define the test plan:
    # plan1: fix 'max_time', if length_of_input_sequence is longer than max_time, drop the oldest element to make the input sequence same length as max_time
    # plan2: use reshape function to reshape input sequence to (max_time, -1)
    test_plan = 2 

    #hyper-paramenter of LSTM cell and RNN
    num_units_in_LSTMCell = n_classes
    max_time =5 
    if test_plan == 1:
        dims_of_input = 1
    else:
        dims_of_input = length_of_input_sequence / max_time 
    lr = 0.01
    n_iteration = 100

    # Define RNN network structure #
    sess = tf.Session()
    with tf.name_scope('net_define'):
        batch_size_t = tf.placeholder(tf.int32, None)
        inputTensor = tf.placeholder(tf.float32, [None, max_time, dims_of_input], name='inputTensor')
        labelTensor = tf.placeholder(tf.float32, [None, n_classes], name='LabelTensor')
        lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units_in_LSTMCell)
        init_state = lstmCell.zero_state(batch_size_t, dtype=tf.float32)
        raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, inputTensor, initial_state = init_state)
        outputs = tf.unstack(tf.transpose(raw_output, [1, 0, 2]), name='outputs_before_softmax')
        output = outputs[-1];
        output = tf.identity(output, 'tensor_before_softmax')
        y_predict = tf.nn.softmax(output, name='softmax_output')
    
    #define evaluation ops
    with tf.name_scope('evaluation'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labelTensor, logits=output)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', mean_cross_entropy)

        correct_pred = tf.equal(tf.argmax(y_predict, 1), tf.argmax(labelTensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    #define train ops
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    #tensorboard
    tensorboard_log_path = 'tf_writer'
    merged = tf.summary.merge_all()
    removeFileInDir(tensorboard_log_path)
    train_writer = tf.summary.FileWriter(tensorboard_log_path + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(tensorboard_log_path + '/test')

    #define saver
    all_vars = tf.global_variables()
    saver = tf.train.Saver(all_vars)

    #generate data and label
    train_data, train_label, test_data,test_label = genTrainAndTestData(n_train_size, n_test_size)
    print "train data sample: ", train_data[0], " Lable=", train_label[0]
    print "train data sample: ", train_data[1], " Lable=", train_label[1]

    #init TF session
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #Train and evaluation
    for iterIdx in range(0, n_iteration):
        for batchIdx in range(int(n_train_size/batch_size)):
            x_train_batch = train_data[batchIdx*batch_size: (batchIdx+1)*batch_size]
            if test_plan == 1:
                x_train_batch = x_train_batch[:,-max_time:,:] 
            else:
                x_train_batch = np.reshape(x_train_batch, [batch_size, max_time, -1])
            y_train_batch = train_label[batchIdx*batch_size: (batchIdx+1)*batch_size]

            sess.run([train_op], feed_dict={
                    inputTensor: x_train_batch,
                    labelTensor: y_train_batch,
                    batch_size_t: batch_size})


            acc, trainloss, summary  = sess.run([accuracy, mean_cross_entropy, merged], feed_dict={
                    inputTensor: x_train_batch,
                    labelTensor: y_train_batch,
                    batch_size_t: batch_size
                        })

        train_writer.add_summary(summary, iterIdx)
        train_writer.flush()

        #print "IterIdx = ", iterIdx, " train_ce = ", trainloss

        if test_plan == 1:
            test_batch = test_data[:,-max_time:,:] 
        else:
            test_batch = np.reshape(test_data, [batch_size, max_time, -1])
        test_acc, test_ce, test_summary = sess.run([accuracy, mean_cross_entropy, merged], feed_dict={inputTensor: test_batch, labelTensor: test_label, batch_size_t: batch_size})

        test_writer.add_summary(test_summary, iterIdx)
        test_writer.flush()

        #print "IterIdx = ", iterIdx, " test_ce = ", test_ce
