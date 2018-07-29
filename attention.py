__author__ = "Yunlong Yu"
__copyright__ = "--"
from utils import *
import tensorflow as tf

class Attention():
    def __init__(self, args, data):
        self.seman_dim = args.seman_dim
        self.hidden_dim = args.hidden_dim
        self.img_dim = args.img_dim
        self.part_num = args.part_num
        self.sub_dim = args.sub_dim
        self.batch_size = args.batch_size
        self.drop_out_rate = args.drop_out_rate
        self.class_num = args.class_num
        self.lr = args.learning_rate
        self.data = data
        self.n_step = args.max_iter

        # attribute-to-img embedding
        self.seman_img_W = tf.Variable(tf.truncated_normal([self.seman_dim, self.sub_dim], stddev=0.01))
        self.seman_img_b = tf.Variable(tf.constant(0.1,shape=[self.sub_dim]))
        # attribute-to-img-att embedding
        self.seman_img_att_W = tf.Variable(tf.truncated_normal([self.seman_dim, self.img_dim],stddev=0.01))
        self.seman_img_att_b = tf.Variable(tf.constant(0.1,shape=[self.img_dim]))
        # img-to-attribute embedding
        self.img_seman_W = tf.Variable(tf.truncated_normal([self.sub_dim, self.seman_dim],stddev=0.01))
        self.img_seman_b = tf.Variable(tf.constant(0.1,shape=[self.seman_dim]))
        # img-att to attribute embedding
        self.att_img_seman_W = tf.Variable(tf.truncated_normal([self.sub_dim,self.seman_dim],stddev=0.01))
        self.att_img_seman_b = tf.Variable(tf.constant(0.1,shape=[self.seman_dim]))

        self.build_model()

    def build_model(self):
        self.image = tf.placeholder(tf.float32, [None, self.img_dim])
        self.seman = tf.placeholder(tf.float32, [None, self.seman_dim])
        self.label = tf.placeholder(tf.float32,[None,self.class_num])
        self.seman_b = tf.placeholder(tf.float32,[None,self.seman_dim])

        # For image
        image_emb = tf.reshape(self.image, [-1, self.part_num, self.sub_dim])  # ? x 7 x 512
        image_feats = tf.reduce_sum(tf.stack(image_emb), 1)  # ? x 512

        with tf.variable_scope("Predict_attribute") as scope:
            pred_seman = tf.nn.relu(tf.matmul(image_feats,self.img_seman_W)+self.img_seman_b)  # The predicted attribute based on the image

        # pred_attr --> attr, loss1 = pred_attr-attr
        # attention_layer 1
        with tf.variable_scope("loss1"):
            loss1 = tf.reduce_sum(tf.square(pred_seman-self.seman))
        att_seman_input = pred_seman
        att_img_input  = tf.reshape(image_emb,[-1,self.sub_dim])

        # Attention model
        with tf.variable_scope("att1"):
            img_att = self.attention(att_seman_input, att_img_input)

        att_img_input = img_att + att_img_input
        att_img = tf.reshape(att_img_input,[-1,self.part_num,self.sub_dim])
        img_att = tf.reduce_sum(tf.stack(att_img),1)

        # #attention_layer 2
        att_seman_input = tf.nn.relu(tf.matmul(img_att,self.img_seman_W)+self.img_seman_b)
        with tf.variable_scope("loss2"):
            loss2 = tf.reduce_sum(tf.square(att_seman_input-self.seman))

        with tf.variable_scope('att2'):
            img_att = self.attention(att_seman_input, att_img_input)
            att_img_input = img_att + att_img_input

        self.img_att = tf.reshape(att_img_input, [-1, self.img_dim])

        # Basic model
        self.seman_image_emb = tf.nn.relu(tf.matmul(self.seman_b, self.seman_img_att_W)+self.seman_img_att_b)
        seman_img_emb = tf.nn.relu(tf.matmul(self.seman,self.seman_img_att_W)+self.seman_img_att_b)
        with tf.variable_scope("Loss") as scope:
            logit = tf.matmul(self.img_att,tf.transpose(self.seman_image_emb))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.label))
            loss3 = tf.reduce_mean(tf.square(seman_img_emb - self.img_att))
            loss = loss1+loss2+ loss3 + loss   #1e0
            # regularizer
            vars = tf.trainable_variables()
            regularizer = tf.add_n([tf.nn.l2_loss(v) for v in vars])
            loss += regularizer

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(loss)
        self.init = tf.global_variables_initializer()

    def train(self):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        tr_img = self.data['train_feats']
        tr_seman = self.data['train_seman']
        tr_label = self.data['train_lb']
        tr_seman_pro = self.data['train_seman_pro']

        with tf.Session() as sess:

            sess.run(self.init)

            for iter in range(self.n_step):
                train_loader = get_batch(tr_img, tr_seman, tr_label, self.batch_size)
                img_batch,seman_batch,label_batch = next(train_loader)
                sess.run(self.optimizer,feed_dict={self.seman:seman_batch,self.image:img_batch,self.seman_b:tr_seman_pro,self.label:label_batch})
                if np.mod(iter,50) == 0:
                    # Test
                    result = test_att_model(sess,self.image,self.seman_b,self.seman_image_emb,self.img_att,self.data)
                    print("The iter %d, the accuracy is: %g" % (iter, result))



    def attention(self,att_seman_input,att_img_input):

        img_origin = tf.reshape(att_img_input,[-1,self.part_num,self.sub_dim])
        # image-attention embedding
        img_att_W = tf.Variable(tf.truncated_normal([self.sub_dim, self.hidden_dim], stddev=0.01))
        img_att_b = tf.Variable(tf.constant(0.1,shape=[self.hidden_dim]))
        # attr-attention embedding
        seman_att_W = tf.Variable(tf.truncated_normal([self.seman_dim,self.hidden_dim],stddev=0.01))
        seman_att_b = tf.Variable(tf.constant(0.1,shape=[self.hidden_dim]))

        # probablity
        prob_att_W = tf.Variable(tf.truncated_normal([self.hidden_dim, 1], stddev=0.01))
        prob_att_b = tf.Variable(tf.constant(0.1,shape=[1]))

        seman_att = tf.expand_dims(att_seman_input,1)
        seman_att = tf.tile(seman_att,tf.constant([1,self.part_num,1]))
        seman_att = tf.reshape(seman_att,[-1,self.seman_dim])
        seman_att = tf.tanh(tf.matmul(seman_att,seman_att_W)+seman_att_b)

        image_att = tf.nn.tanh(tf.matmul(att_img_input,img_att_W)+img_att_b)

        output_att = tf.tanh(image_att+seman_att) # tanh 

        prob_att = tf.matmul(output_att,prob_att_W)+prob_att_b
        prob_att = tf.reshape(prob_att,[-1,self.part_num])
        prob_att = tf.nn.softmax(prob_att)   # 

        with tf.variable_scope('image_out') as scope:
            image_att = tf.reshape(tf.multiply(img_origin,tf.expand_dims(prob_att,2)),[-1,self.sub_dim])

        return image_att
