import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_glas import GLAS_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/GLAS/test_vol_npz', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='GLAS', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_GLAS', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=250, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-5, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def inference(args, model, test_save_path=None):
    # Ensure the correct Dataset class is used (already done in __main__)
    try:
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir) # Use the correct split name
        # Ensure 'test_vol' exists in the list directory for GLAS
    except FileNotFoundError:
         # Try common alternative names if 'test_vol.txt' is not found
         found_split = None
         common_splits = ["test", "validation", "test_list"]
         for split_name in common_splits:
             try:
                 list_file = os.path.join(args.list_dir, f"{split_name}.txt")
                 if os.path.exists(list_file):
                     db_test = args.Dataset(base_dir=args.volume_path, split=split_name, list_dir=args.list_dir)
                     logging.info(f"Using split file: {list_file}")
                     found_split = True
                     break
             except FileNotFoundError:
                 continue
         if not found_split:
             logging.error(f"Could not find a suitable test list file in {args.list_dir} (tried test_vol.txt, test.txt, validation.txt, test_list.txt)")
             return "Testing Failed: Test list not found."
    except Exception as e:
        logging.error(f"Error initializing dataset: {e}")
        return "Testing Failed: Dataset initialization error."


    # Use batch_size=1 for testing individual 2D images
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} testing images.") # Changed logging message slightly
    model.eval() # Set model to evaluation mode

    # Use a list to collect metrics per image, then convert to numpy array
    all_metrics = []
    num_classes_for_metric = args.num_classes # Use this for metric calculation range (includes background)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        # Image should be (B, C, H, W), Label should be (B, H, W) -> with B=1
        # case_name is a tuple of strings if batch_size > 1, accessing [0] is correct for B=1
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # Call the modified test_single_volume (which now handles 2D)
        # No z_spacing needed
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name)

        if metric_i is not None: # Check if metric calculation was successful
             all_metrics.append(metric_i) # Add the list of class metrics for this image
             # Log metrics for the current image (optional, can be verbose)
             # Example: Log average Dice/HD95 for this image across foreground classes
             if len(metric_i) > 0: # Ensure there are foreground class metrics
                 img_mean_dice = np.mean([m[0] for m in metric_i])
                 img_mean_hd95 = np.mean([m[1] for m in metric_i])
                 logging.info(f'idx {i_batch} case {case_name} mean_dice {img_mean_dice:.4f} mean_hd95 {img_mean_hd95:.4f}')
             else:
                 logging.info(f'idx {i_batch} case {case_name} - No foreground classes to evaluate or metrics returned empty.')
        else:
             logging.warning(f"Metrics calculation failed for case {case_name}. Skipping accumulation for this case.")

    # After iterating through all images, calculate overall average metrics
    if not all_metrics: # Check if any metrics were collected
        logging.warning("No valid metrics were collected during testing.")
        return "Testing Finished - No metrics."

    # Convert the list of lists into a numpy array: (num_images, num_foreground_classes, 2) -> (dice, hd95)
    # Example: if num_classes=2, metric_list shape is (num_images, 1, 2)
    # Example: if num_classes=4, metric_list shape is (num_images, 3, 2)
    try:
        metric_array = np.array(all_metrics)
        num_images_evaluated = metric_array.shape[0]
        num_fg_classes = metric_array.shape[1] # Should be args.num_classes - 1

        # Calculate mean metrics per foreground class across all images
        mean_metrics_per_class = np.mean(metric_array, axis=0) # Shape: (num_fg_classes, 2)

        for i in range(num_fg_classes):
            class_idx = i + 1 # Actual class index (1, 2, ...)
            mean_dice = mean_metrics_per_class[i, 0]
            mean_hd95 = mean_metrics_per_class[i, 1]
            logging.info(f'Mean metrics for class {class_idx}: mean_dice {mean_dice:.4f}, mean_hd95 {mean_hd95:.4f}')

        # Calculate overall performance: average Dice and HD95 across all foreground classes and all images
        overall_mean_dice = np.mean(mean_metrics_per_class[:, 0])
        overall_mean_hd95 = np.mean(mean_metrics_per_class[:, 1])

        logging.info(f'Overall Testing Performance ({num_images_evaluated} images):')
        logging.info(f'Mean Dice (across foreground classes): {overall_mean_dice:.4f}')
        logging.info(f'Mean HD95 (across foreground classes): {overall_mean_hd95:.4f}')

    except Exception as e:
        logging.error(f"Error calculating final average metrics: {e}")
        return "Testing Finished - Error in metric aggregation."


    return "Testing Finished!"


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

    dataset_config = {
        'GLAS': {
            'Dataset': GLAS_dataset,
            'volume_path': '../data/GLAS/test_vol_npz',
            'list_dir': './lists/lists_GLAS',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


