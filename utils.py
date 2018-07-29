import tensorflow as tf
import scipy.io as sio
from numpy import *
import numpy as np


def load_data(db_name,data_source,feat_model):
    dir = data_source+db_name+'_text_pca_'+feat_model
    f = sio.loadmat(dir)
    train_feats = np.array(f['train_feats'])
    train_seman = np.array(f['train_seman'])

    train_seman_pro = np.array(f['train_seman_pro'])
    test_feats = np.array(f['test_feats'])
    test_seman_pro = np.array(f['test_seman_pro'])
    test_label = np.array(f['test_label'])
    testclasses_id = np.array(f['testclasses_id'])
    train_lb = np.array(f['tr_gt_label'])

    data = {
        'train_feats': train_feats,
        'train_seman': train_seman,
        'train_seman_pro':train_seman_pro,
        'train_lb': train_lb,
        'test_feats':test_feats,
        'test_seman_pro':test_seman_pro,
        'test_label':test_label,
        'testclasses_id':testclasses_id
    }
    return data

def get_batch(img,attr,label,batch_size):
    while True:
        idx = np.arange(0,len(img))
        np.random.shuffle(idx)
        shuf_visual = img[idx]
        shuf_attr = attr[idx]
        shuf_label = label[idx]

        for batch_index in range(0,len(img),batch_size):
            visual_batch = shuf_visual[batch_index:batch_index+batch_size]
            attr_batch = shuf_attr[batch_index:batch_index+batch_size]
            label_batch = shuf_label[batch_index:batch_index+batch_size]
            yield visual_batch, attr_batch, label_batch

def cosine_distance(v1,v2):
    v1_sq = np.inner(v1,v1)
    v2_sq = np.inner(v2,v2)
    dis = 1 - np.inner(v1,v2)/math.sqrt(v1_sq*v2_sq)
    return dis

# classify using kNN
def kNNClassify(newInput,dataSet,labels,k):
    global distance
    distance = [0] * dataSet.shape[0]
    for i in range(dataSet.shape[0]):
        distance[i] = cosine_distance(newInput,dataSet[i])

    # sort the distance
    sortedDisIndices = np.argsort(distance)
    classCount = {}  # difine a dictionary

    for i in range(k):
        voteLabel = labels[sortedDisIndices[i]]

        classCount[voteLabel] = classCount.get(voteLabel,0) + 1

    # the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


def Distance(newInput,dataSet):
    global cosine_distance
    distance = [0] * dataSet.shape[0]
    for i in range(dataSet.shape[0]):
        distance[i] = cosine_distance(newInput, dataSet[i])

    return distance


def test_att_model(sess,image,seman,seman_image_emb,img_att,data):
    test_seman_pro = data['test_seman_pro']
    testclasses_id = data['testclasses_id']
    test_label = data['test_label']
    test_feats = data['test_feats']
    test_num = len(test_feats)

    att_pre = sess.run(seman_image_emb, feed_dict={seman: test_seman_pro})
    # att_pre = regularization(att_pre)
    test_label = np.squeeze(np.asarray(test_label))
    test_label = test_label.astype("float32")

    img_att = sess.run(img_att, feed_dict={image: test_feats})
    # img_att = regularization(img_att)
    test_id = np.squeeze(np.asarray(testclasses_id))
    outpre = [0] * test_num  # CUB 2933
    test_label = np.squeeze(np.asarray(test_label))
    test_label = test_label.astype("float32")

    Distance_mat = []
    for i in range(test_num):  # CUB 2933
        distance = Distance(img_att[i, :], att_pre)
        Distance_mat.append(distance)
    Distance_mat = np.array(Distance_mat)

    for j in range(test_num):
        distance = Distance_mat[j, :]
        sortedDisIndices = np.argsort(distance)
        classCount = {}  # difine a dictionary

        for i in range(1):
            voteLabel = test_id[sortedDisIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        # the max voted class will return
        maxCount = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        outpre[j] = maxIndex.astype("int32")
    correct_prediction = tf.equal(outpre, test_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={seman: test_seman_pro, image: test_feats})
    return result








