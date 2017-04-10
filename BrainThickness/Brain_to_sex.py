import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
path="C:/Users/cha/Desktop/brain/"

def logireg (X1,Y1,X2,Y2, lr,epoch, dr, batch_size, filename, decaying_rate):
    cols=X1.shape[1]
    train_n=X1.shape[0]
    x=tf.placeholder(tf.float32, shape=[None,cols])
    y=tf.placeholder(tf.float32, shape=[None,1])
    dropout_rate=tf.placeholder(tf.float32)
    decaying_lr=tf.placeholder(tf.float32)

    # 3 hidden layers
    w1=tf.get_variable('w1',shape=[cols,256], initializer=tf.contrib.layers.xavier_initializer())
    w2=tf.get_variable('w2',shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
    w3=tf.get_variable('w3',shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
    w4=tf.get_variable('w4',shape=[256,1], initializer=tf.contrib.layers.xavier_initializer())

    b1=tf.Variable(tf.random_normal([256]), name='bias1')
    b2=tf.Variable(tf.random_normal([256]), name='bias2')
    b3=tf.Variable(tf.random_normal([256]), name='bias3')
    b4=tf.Variable(tf.random_normal([1]), name='bias3')

    # 3 hidden layers with ReLUs as activation functions and using dropout method
    _L1=tf.nn.relu(tf.matmul(x,w1)+b1)
    L1=tf.nn.dropout(_L1, dropout_rate)
    _L2=tf.nn.relu(tf.matmul(L1,w2)+b2)
    L2=tf.nn.dropout(_L2, dropout_rate)
    _L3=tf.nn.relu(tf.matmul(L2,w3)+b3)
    L3=tf.nn.dropout(_L3, dropout_rate)

    hypothesis=tf.sigmoid(tf.matmul(L3,w4) + b4)

    # Using Adam optimizer
    cost=-tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
    train=tf.train.AdamOptimizer(learning_rate=decaying_lr).minimize(cost)

    predicted=tf.cast(hypothesis>0.5, dtype=tf.float32)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

    # For saving trained model
    tf.add_to_collection('vars',w1)
    tf.add_to_collection("vars",w2)
    tf.add_to_collection('vars',w3)
    tf.add_to_collection("vars",w4)
    tf.add_to_collection("vars",b1)
    tf.add_to_collection("vars",b2)
    tf.add_to_collection("vars",b3)
    tf.add_to_collection("vars",b4)
    saver=tf.train.Saver()

    if train_n%100==0:
        total_batch=int(train_n/batch_size)
    else:
        total_batch=int(train_n/batch_size)+1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Training
        for step in range(epoch):
            for j in range(total_batch):
                if (j+1)*batch_size <= train_n:
                    batch_x=X1[j*batch_size:(j+1)*batch_size,:]
                    batch_y=Y1[j*batch_size:(j+1)*batch_size,:]
                else:
                    batch_x=X1[j*batch_size:,:]
                    batch_y=Y1[j*batch_size:,:]
                cost_val, _ = sess.run([cost,train], feed_dict={x: batch_x, y: batch_y, dropout_rate:dr, decaying_lr: lr})
            # Showing loss values with decaying learning rate
            if (step+1)%100==0:
                print(step+1,cost_val)
                saver.save(sess, path+filename)
                lr=lr*decaying_rate**((step+1)/100)
            # Showing interim accuracy
            if (step+1)%500==0:
                mid_a = sess.run([accuracy], feed_dict={x: X2, y: Y2, dropout_rate: 1.0})
                print("Accuracy: ", mid_a)
            # Break if loss is NaN
            if np.isnan(cost_val):
                # Exit the function if loss is NaN at early stage
                if step+1 < 1000:
                    print("Use lower learning rate")
                    return 0
                print ("Total steps: ", step+1)
                break
        saver.save(sess,path+filename)
        # accuracy report
        h,c,a=sess.run([hypothesis, predicted, accuracy], feed_dict={x:X2, y:Y2, dropout_rate:1.0})
        print("\n Hypothesis: ", h, "\nCorrect (Y) ", c, "\nAccuracy: ", a)
#x=scaler.fit_transform(np.loadtxt(path+'hcpThick731.csv', delimiter=",", dtype=np.float32))
x=np.loadtxt(path+'hcpThick731.csv', delimiter=",", dtype=np.float32)
sex=np.loadtxt(path+'hcpGender.csv', delimiter=",", dtype=np.float32)
gender=np.where(sex==1, 0, 1)
train_x=x.T[0:657,:]
train_y=np.array([gender]).T[0:657,:]
test_x=x.T[657:,:]
test_y=np.array([gender]).T[657:,:]

logireg(train_x, train_y, test_x, test_y,lr=1e-6,epoch=20000,dr=0.7, batch_size=200, filename="Gender_1e-6", decaying_rate=0.99)



