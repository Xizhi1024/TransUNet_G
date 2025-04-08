import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_GLAS
#without any change
#21:57 train_time 5.27s/it
# 22%|██████▉                         | 54/250 [04:42<17:07,  5.24s/it]
#iteration 1189 : loss : 0.071356, loss_ce: 0.108342
#iteration 1190 : loss : 0.063878, loss_ce: 0.096933
#Mean Dice (across foreground classes): 0.9088
#Mean HD95 (across foreground classes): 46.3345
#  transformer    :   97,888,576 (97.89M)
#  decoder        :    7,390,089 (7.39M)
#  segmentation_head:          290 (0.00M)
# 总参数量: 105,278,955 (105.28M)


# 22:08<00:05,  5.33s/it]
# 24%|███████▌                        | 59/250 [05:11<16:46,  5.27s/it]
#iteration 1299 : loss : 0.056412, loss_ce: 0.086225
#iteration 1300 : loss : 0.080967, loss_ce: 0.122705
#Mean Dice (across foreground classes): 0.9119
#Mean HD95 (across foreground classes): 43.7279
#  transformer    :   97,888,576 (97.89M)
#  decoder        :    7,421,065 (7.42M)
# segmentation_head:          290 (0.00M)
#总参数量: 105,309,931 (105.31M)

#[22:36<00:05,  5.45s/it]
# 21%|██████▊                         | 53/250 [04:48<17:47,  5.42s/it]
# iteration 1168 : loss : 0.070937, loss_ce: 0.108260
# iteration 1169 : loss : 0.074162, loss_ce: 0.113169
#Mean Dice (across foreground classes): 0.9093
#Mean HD95 (across foreground classes): 49.1621
#   transformer    :   97,888,576 (97.89M)
#   decoder        :    7,421,065 (7.42M)
#   segmentation_head:          290 (0.00M)
#   fsb            :      981,156 (0.98M)
#   aam            :    1,182,721 (1.18M)

# 总参数量: 107,473,808 (107.47M)








parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/GLAS/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='GLAS', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_GLAS', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=250, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-5,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'GLAS': {
            'root_path': '../data/GLAS/train_npz',
            'list_dir': './lists/lists_GLAS',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'GLAS': trainer_GLAS,}
    trainer[dataset_name](args, net, snapshot_path)