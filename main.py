# Main driver file

import dataset
import os
import TransFM
import FM
import PRME_FM
import HRM_FM
import sys
import argparse
import tensorflow as tf

# config
filename        = 'ratings.csv' 
model           = 'TransFM'  
features        = 'none' 
features_file   = 'none' 
max_iters       = '1000000' 
num_dims        = '10' 
linear_reg      = '10.0' 
emb_reg         = '1.0'
trans_reg       = '0.1' 
init_mean       = '0.1' 
starting_lr     = '0.02' 
lr_decay_factor = '1.0' 
lr_decay_freq   = '1000' 
eval_freq       = '50' 
quit_delta      = '1000'

print(sys.argv)


def parse_args( filename,     model,            features,       features_file,  max_iters,       
                num_dims,     linear_reg,       emb_reg,        trans_reg,      init_mean,     
                starting_lr,  lr_decay_factor,  lr_decay_freq,  eval_freq,      quit_delta ):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
        help='Filename of the input dataset.',
        required=True)
    parser.add_argument('--model',
        help='Model to run.',
        choices=['TransFM', 'FM', 'PRME-FM', 'HRM-FM'],
        required=True)
    parser.add_argument('--features',
        help='Which features to include.',
        choices=['none', 'categories', 'time', 'content', 'geo'],
        default='none')
    parser.add_argument('--features_file',
        help='Filename(s) for content features. For content features, provide '
        '<user filename>,<item filename>. For categories and geo, provide a single '
        'filename. Temporal data should be included within the dataset file itself.')
    parser.add_argument('--max_iters',
        help='Max number of iterations to run',
        default=1000000,
        type=int)
    parser.add_argument('--num_dims',
        help='Model dimensionality.',
        default=10,
        type=int)
    parser.add_argument('--linear_reg',
        help='L2 regularization: linear_reg.',
        default=1.0,
        type=float)
    parser.add_argument('--emb_reg',
        help='L2 regularization: embbeding regularization.',
        default=1.0,
        type=float)
    parser.add_argument('--trans_reg',
        help='L2 regularization: translation regularization.',
        default=1.0,
        type=float)
    parser.add_argument('--init_mean',
        help='Initialization mean for model parameters.',
        default=0.1,
        type=float)
    parser.add_argument('--starting_lr',
        help='Initial learning rate.',
        default=0.001,
        type=float)
    parser.add_argument('--lr_decay_factor',
        help='Decay factor for learning rate.',
        default=1.0,
        type=float)
    parser.add_argument('--lr_decay_freq',
        help='Frequency at which to decay learning rate.',
        default=1000,
        type=int)
    parser.add_argument('--eval_freq',
        help='Frequency at which to evaluate model.',
        default=50,
        type=int)
    parser.add_argument('--quit_delta',
        help='Number of iterations at which to quit if no improvement.',
        default=1000,
        type=int)
    args = parser.parse_args(args = [ 
                            '--filename',        filename,
                            '--model',           model,
                            '--features',        features,
                            '--features_file',   features_file, 
                            '--max_iters',       max_iters,
                            '--num_dims',        num_dims,
                            '--linear_reg',      linear_reg,
                            '--emb_reg',         emb_reg,
                            '--trans_reg',       trans_reg,
                            '--init_mean',       init_mean,
                            '--starting_lr',     starting_lr,
                            '--lr_decay_factor', lr_decay_factor,
                            '--lr_decay_freq',   lr_decay_freq,
                            '--eval_freq' ,      eval_freq,
                            '--quit_delta' ,     quit_delta
                    ])
    print(args)
    print('')
    return args

def train_transrec(dataset, args):
    if args.model == 'TransFM':
        model = TransFM.TransFM(dataset, args)
    elif args.model == 'FM':
        model = FM.FM(dataset, args)
    elif args.model == 'PRME-FM':
        model = PRME_FM.PRME_FM(dataset, args)
    elif args.model == 'HRM-FM':
        model = HRM_FM.HRM_FM(dataset, args)

    val_auc, test_auc,  var_emb_factors, var_trans_factors, g = model.train()

    print('')
    print(args)
    print('Validation AUC  = ' + str(val_auc))
    print('Test AUC        = ' + str(test_auc))
    return (val_auc, test_auc,  var_emb_factors, var_trans_factors, g)


if __name__ == '__main__':
    args = parse_args(  filename,     model,            features,       features_file,  max_iters,       
                        num_dims,     linear_reg,       emb_reg,        trans_reg,      init_mean,     
                        starting_lr,  lr_decay_factor,  lr_decay_freq,  eval_freq,      quit_delta )
    d = dataset.Dataset(args.filename, args)
    train_transrec(d, args)