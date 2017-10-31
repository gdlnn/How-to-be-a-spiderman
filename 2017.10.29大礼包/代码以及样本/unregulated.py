import tensorflow as tf
import numpy as np
import loadfun as lf

def add_a_layer(x,in_dim,out_dim,act_fun=None,mod=0):
    W=tf.Variable(tf.random_normal([in_dim,out_dim],stddev=1.0))#设置权重
    B=tf.Variable(tf.zeros([1,out_dim]))#设置偏正值
    xW_plus_B=tf.matmul(x,W)+B#计算xW+B的值
    if mod == 1:
        xW_plus_B=tf.div(xW_plus_B,1000)
    if act_fun == None:
        return xW_plus_B
    else:
        return act_fun(xW_plus_B)

def main(learing_rate,train_length):
    
    x_data=tf.placeholder(tf.float32,[None,784])#设置placeholder，为x_data变量保留一块空间，之后调用，下同
    y_data=tf.placeholder(tf.float32,[None,10])
    
    HL1O=add_a_layer(x_data,784,28,tf.nn.sigmoid,1)#这里的28可以改成10，这里之所以可以是20是因为下面还有第二个隐藏单元可以把输出结果变成10
    #可以把上面的HL1O改成HL11O，然后把下面三句前面的#删掉
    #HL12O=add_a_layer(x_data,784,28,tf.nn.sigmoid,1)
    #HL1O=tf.multiply(HL11O,HL12O)
    #HL1O=tf.pow(HL1O,0.5)
    prediction=add_a_layer(HL1O,28,10,tf.nn.softmax)
    loss=-tf.reduce_sum(y_data*tf.log(prediction))
    train_step=tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)#优化器，告诉tensorflow我们最小化的目标是loss，学习率为main函数的参数learning_rate，大家可以修改学习率看看效果
    init=tf.global_variables_initializer()#初始化，这里的init同样是个tensor，意味着要通过run函数来执行它
    
    with tf.Session() as sess:
        sess.run(init)
        train_IMG=lf.loadImageSet("train-images.idx3-ubyte")
        train_LAB=lf.loadLabelSet("train-labels.idx1-ubyte")
        
        for i in range(train_length):
            for j in range(60000):
                res=np.zeros([1,10])#这一句和下一句用于制作one-hot vector
                res[0,train_LAB[j,0]]=1
                in_IMG=np.reshape(train_IMG[j,:],[1,784])
                sess.run(train_step,feed_dict={x_data:in_IMG,y_data:res})#注意这里feed_dict的用法，feed_dict用于给之前设置好的placeholder填值
                
        print("training done")
        
        test_IMG=lf.loadImageSet("t10k-images.idx3-ubyte")
        test_LAB=lf.loadLabelSet("t10k-labels.idx1-ubyte")
        
        right=0
        for k in range(10000):
            in_IMG=np.reshape(test_IMG[k,:],[1,784])
            res=sess.run(prediction,feed_dict={x_data:in_IMG,y_data:np.zeros([1,10])})
            if res[0].argmax() == test_LAB[k,0]:
                right+=1
            print("prediction:",res[0].argmax(),"real answer:",test_LAB[k,0])
            print(right/(k+1))
            
main(0.1,1)