__author__ = "Yunlong Yu"
__copyright__ = "--"
import argparse
from attention import Attention
from utils import *

if __name__=='__main__':

    db_name = 'nabird'
    data_source = './data/'

    # Model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_model',type=str,default='easy',help='easy or hard')
    parser.add_argument('--seman_dim', type=int, default=400, help='the semantic dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden dimension') #128
    parser.add_argument('--img_dim', type=int, default=3072, help='the visual dimension')
    parser.add_argument('--part_num',type=int,default=6,help='the part number')
    parser.add_argument('--sub_dim',type=int,default=512,help='the dimension of each part')
    parser.add_argument('--class_num', type=int, default=323, help='the training class number')
    parser.add_argument('--batch_size',type=int,default=1024,help='the batch size') #64
    parser.add_argument('--drop_out_rate',type=float,default=0.5,help='drop out rate')
    parser.add_argument('--max_iter',type=int,default=3000,help='the number of max iteration')
    parser.add_argument('--learning_rate',type=float,default=0.0002,help='the learning rate, default is 1e-3')
    args = parser.parse_args()

    # setup the dataset
    dataset = load_data(db_name,data_source,args.feat_model)

    model = Attention(args,dataset)
    model.train()


