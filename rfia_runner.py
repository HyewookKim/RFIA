import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *

import copy


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='C:/Users/hyewo/Desktop/VScode/Image_set', help='Path to the datasets directory. Default is ./dataset/')
    # parser.add_argument('--data-root', dest='data_root', type=str, default='/root/Image_set', help='Path to the datasets directory. Default is ./dataset/')    
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, item, shot_capacity, num_count, clip_weights,representation_cache,sub_rep_cache):    
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        # item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]     
        if pred in cache:
            if len(cache[pred]) < shot_capacity: 
                cache[pred].append(item)
                num_count[pred] = num_count[pred] + 1
                representation_item = representation_cache[pred][0]
                representation_item[0] = modify_representation_feature(representation_item,item,num_count[pred])
                sub_rep_cache[pred] = get_sub_representation_feature(representation_item,cache[pred])
                representation_item[0] = compare_representation_feature(representation_item[0], sub_rep_cache[pred][0][0], pred, clip_weights)                
                            
            # elif features_loss[1] < cache[pred][-1][1]:
            elif item[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
                num_count[pred] = num_count[pred] + 1
                representation_item = representation_cache[pred][0]
                representation_item[0] = modify_representation_feature(representation_item,item,num_count[pred])
                sub_rep_cache[pred] = get_sub_representation_feature(representation_item,cache[pred])
                representation_item[0] = compare_representation_feature(representation_item[0], sub_rep_cache[pred][0][0], pred, clip_weights)
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
            num_count[pred] = 0
            representation_item = copy.deepcopy(item)
            representation_item[1] = torch.zeros_like(item[1]) # item[1] is entropy. However, representation_item do not use entropy. So We initialize loss value = 0 for convenience
            representation_cache[pred] = [representation_item]           

    
def compute_cache_logits(image_features, cache, representation_cache, sub_rep_cache, alpha, beta, clip_weights, device):
    """Compute logits using positive/negative cache.""" 
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        merged_cache = merge_dictionary(cache, sub_rep_cache)           
        for class_index in sorted(cache.keys()):
            merged_cache[class_index] = [representation_cache[class_index][0]] + [representation_cache[class_index][0]] + merged_cache[class_index][:]
            for item in merged_cache[class_index]:
                cache_keys.append(item[0])
                cache_values.append(class_index)                  
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).to(device).half()
        affinity = image_features @ cache_keys
        cache_values = cache_values.to(dtype=affinity.dtype)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits   

def run_test_rfia(pos_cfg, loader, clip_model, clip_weights, wandb_test, device):
    with torch.no_grad():
        pos_cache, accuracies = {}, []
        representation_cache = {}
        sub_rep_cache = {}
        pos_num_count = {}

        #Unpack all hyperparameters
        pos_enabled = pos_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}

        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights, device)
            target = target.to(device)
            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'], pos_num_count, clip_weights,
                                 representation_cache, sub_rep_cache)
                # update_cache(pos_cache, pred, [image_features, loss, prob_map], pos_params['shot_capacity'], pos_num_count, clip_weights,
                #                  representation_cache, sub_rep_cache)
            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache,representation_cache,sub_rep_cache, pos_params['alpha'], pos_params['beta'], clip_weights, device)

            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            if wandb_test :
                wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)
            # if i%1000==0:
            #     print("---- RFIA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))              
            if i%100==0:
                print("---- RFIA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
            if i==200:
                break
        print("---- RFIA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        return sum(accuracies)/len(accuracies)    



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    device = next(clip_model.parameters()).device
    print(f"Using device: {device}")

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    args.wandb = False
    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run RFIA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model, device)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="RFIA", config=cfg, group=group_name, name=run_name)

        acc = run_test_rfia(cfg['positive'], test_loader, clip_model, clip_weights, args.wandb, device)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()