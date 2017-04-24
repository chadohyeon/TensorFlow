import numpy as np
import tensorflow as tf
path="/home/byeonguk/Downloads/cha/"

lr = 5e-5
batch_size=200
filename="Intel_reg_"+str(lr)
decaying_rate = 0.98
epoch = 10000

x_raw = np.loadtxt(path + 'hcpThick731.csv', delimiter=",", dtype=np.float32).T
intel=np.loadtxt(path+'hcpIntel.csv', delimiter=",", dtype=np.float32)
y_raw=np.array([intel]).T
#for fold in range(10):
fold = 1
X1 = np.append(x_raw[:73 * fold, :], x_raw[73 * (fold + 1):, :], axis=0)
Y1 = np.append(y_raw[:73 * fold, :], y_raw[73 * (fold + 1):, :], axis=0)
X2 = x_raw[73 * fold:73 * (fold + 1), :]
Y2 = y_raw[73 * fold:73 * (fold + 1), :]

tf.reset_default_graph()
cols=X1.shape[1]
train_n=X1.shape[0]
x=tf.placeholder(tf.float32, shape=[None,cols])
y=tf.placeholder(tf.float32, shape=[None,1])

decaying_lr=tf.placeholder(tf.float32)

w1=tf.get_variable('w1',shape=[cols,1], initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([1]), name='bias1')

hypothesis=tf.matmul(x,w1) + b1

# Using Adam optimizer
cost=tf.reduce_mean(tf.square(hypothesis-y))
train=tf.train.AdamOptimizer(learning_rate=decaying_lr).minimize(cost)


# For saving trained model
tf.add_to_collection('vars',w1)
tf.add_to_collection('vars',b1)

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
            cost_val, _ = sess.run([cost,train], feed_dict={x: batch_x, y: batch_y, decaying_lr: lr})
        if (step+1)%100==0:
            saver.save(sess, path+filename+"_"+str(fold))
            lr=lr*decaying_rate**((step+1)/100)
            mse=sess.run([cost], feed_dict={x: X2, y: Y2})
            print("step:" ,step+1," loss:",cost_val," MSE:",mse)

        # Break if loss is NaN
        if np.isnan(cost_val):
            print("Total steps: ", step+1)
            print("Latest saved steps: ", int((step+1)/100)*100)
            print("Latest saved MSE: ", mse)
            break
    saver.save(sess,path+filename+"_"+str(fold))
    mse_total=sess.run([cost], feed_dict={x: X2, y: Y2})
    if fold==1:
      h=sess.run([hypothesis], feed_dict={x: X2, y: Y2})
      result=np.append(h, Y2, axis=1)
      print(result)
    print("Total MSE: ", mse_total)
    
