import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# utils.py / or wherever test_single_volume is defined

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import logging # Added for logging inside function

# --- DiceLoss class remains the same ---
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # Ensure target is on the same device as inputs
        target = target.to(inputs.device)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        # Make sure loss is computed correctly even if weights are used
        # return loss / self.n_classes # Original was potentially wrong if weights != 1
        return loss / sum(weight) if sum(weight) != 0 else loss


def calculate_metric_percase(pred, gt):
    """
    Calculate Dice and HD95 for a single class between prediction and ground truth.
    Assumes pred and gt are binary masks (0 or 1).
    """
    # Ensure inputs are binary
    pred = np.asarray(pred > 0, dtype=np.uint8)
    gt = np.asarray(gt > 0, dtype=np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        try:
            # Use MedPy for Dice and HD95
            dice = metric.binary.dc(pred, gt)
            # Ensure inputs are suitable for hd95 (e.g., not completely empty)
            # hd95 might raise errors if one image is empty or if foreground pixels are too sparse/isolated
            # It also requires the images to have the same dimensions, which should be true here.
            hd95 = metric.binary.hd95(pred, gt) # Voxel spacing defaults to 1x1x...
            return dice, hd95
        except Exception as e:
            logging.warning(f"Could not calculate metrics (Dice/HD95): {e}. Returning 0, 0.")
            return 0.0, 0.0 # Return zeros if metric calculation fails
    elif pred.sum() > 0 and gt.sum() == 0:
        # Prediction has foreground, GT is empty
        return 0.0, 0.0 # Or handle as appropriate (e.g., some fixed penalty for FP, HD95 undefined)
    elif pred.sum() == 0 and gt.sum() > 0:
         # Prediction is empty, GT has foreground
        return 0.0, 0.0 # Or handle as appropriate (e.g., some fixed penalty for FN, HD95 undefined)
    else: # Both pred and gt are empty
        return 1.0, 0.0 # Perfect match (Dice=1), HD95 is typically 0 in this case


# Modified for 2D Input
def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None):
    """
    Tests a single 2D image.
    Args:
        image (torch.Tensor): Input image tensor (1, C, H, W).
        label (torch.Tensor): Ground truth label tensor (1, H, W).
        net (nn.Module): The segmentation network.
        classes (int): Number of output classes.
        patch_size (list): Target patch size for network input [H, W].
        test_save_path (str, optional): Path to save predictions. Defaults to None.
        case (str, optional): Name of the case for saving. Defaults to None.
    Returns:
        list: A list of metrics [(dice, hd95)] for each foreground class.
              Returns None if an error occurs during processing.
    """
    try:
        # Assuming input image is (1, C, H, W) and label is (1, H, W) from DataLoader
        # No need to squeeze batch dim if batch_size=1 in DataLoader
        if image.ndim == 3:
            # Add channel dimension: (B, H, W) -> (B, 1, H, W)
            image = image.unsqueeze(1)
        input_image_tensor = image.cuda() # Move input image to GPU
        input_image_tensor = input_image_tensor.float()
        gt_label_numpy = label.squeeze(0).cpu().detach().numpy() # Label for comparison (H, W)

        # Get original size
        _, _, h, w = input_image_tensor.shape

        # Prepare image for network input (resizing if necessary)
        image_resized = input_image_tensor
        if h != patch_size[0] or w != patch_size[1]:
             # Use torch.nn.functional.interpolate for resizing tensors
             image_resized = nn.functional.interpolate(input_image_tensor, size=patch_size, mode='bilinear', align_corners=False)
             # print(f"Resized image from {h}x{w} to {patch_size[0]}x{patch_size[1]}") # Debugging

        # Inference
        net.eval()
        with torch.no_grad():
            outputs = net(image_resized) # (1, num_classes, patch_size[0], patch_size[1])
            out_softmax = torch.softmax(outputs, dim=1) # (1, num_classes, H, W)
            out_argmax = torch.argmax(out_softmax, dim=1) # (1, H, W)

        # Resize prediction back to original size if necessary
        prediction_resized = out_argmax # Prediction at patch_size resolution

        if h != patch_size[0] or w != patch_size[1]:
            # Resize prediction back to original size (H, W) using nearest neighbor
             prediction_original_size = nn.functional.interpolate(prediction_resized.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1)
             # print(f"Resized prediction from {patch_size[0]}x{patch_size[1]} back to {h}x{w}") # Debugging
        else:
             prediction_original_size = prediction_resized

        # Convert final prediction to numpy array (H, W)
        prediction_numpy = prediction_original_size.squeeze(0).cpu().detach().numpy()

        # Calculate metrics for each foreground class
        metric_list = []
        # print(f"Unique labels in GT: {np.unique(gt_label_numpy)}, Unique labels in Pred: {np.unique(prediction_numpy)}") # Debugging
        for i in range(1, classes): # Iterate from class 1 up to classes-1
            # Create binary masks for the current class
            pred_i = (prediction_numpy == i)
            gt_i = (gt_label_numpy == i)
            # print(f"Class {i}: GT sum={gt_i.sum()}, Pred sum={pred_i.sum()}") # Debugging

            # Calculate metrics for this class
            metric_i = calculate_metric_percase(pred_i, gt_i)
            # print(f"Class {i}: Metrics (Dice, HD95) = {metric_i}") # Debugging
            metric_list.append(metric_i)

        # Save predictions if path is provided
        if test_save_path is not None and case is not None:
            # Ensure prediction is uint8 or similar for image saving
            prediction_to_save = prediction_numpy.astype(np.uint8)
            gt_to_save = gt_label_numpy.astype(np.uint8)
            img_to_save = input_image_tensor.squeeze(0).cpu().detach().numpy() # (C, H, W)

            # Convert to SimpleITK images (handles 2D)
            # For image, potentially take first channel if grayscale, or handle C > 1 if needed
            # Assuming single channel or saving only first channel if C>1
            if img_to_save.shape[0] > 1:
                logging.warning(f"Saving only the first channel of the input image for case {case}.")
                img_itk = sitk.GetImageFromArray(img_to_save[0, :, :].astype(np.float32))
            else:
                 img_itk = sitk.GetImageFromArray(img_to_save.squeeze().astype(np.float32)) # Remove C dim if 1

            prd_itk = sitk.GetImageFromArray(prediction_to_save.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(gt_to_save.astype(np.float32))

            # Set 2D spacing (optional, defaults to 1x1)
            img_itk.SetSpacing((1, 1))
            prd_itk.SetSpacing((1, 1))
            lab_itk.SetSpacing((1, 1))

            # Define filenames
            pred_filename = os.path.join(test_save_path, f"{case}_pred.nii.gz")
            img_filename = os.path.join(test_save_path, f"{case}_img.nii.gz")
            gt_filename = os.path.join(test_save_path, f"{case}_gt.nii.gz")

            # Write images
            sitk.WriteImage(prd_itk, pred_filename)
            sitk.WriteImage(img_itk, img_filename)
            sitk.WriteImage(lab_itk, gt_filename)
            # logging.info(f"Saved prediction, image, and GT for case {case} to {test_save_path}")

        return metric_list

    except Exception as e:
        logging.error(f"Error during testing case {case}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None # Indicate failure