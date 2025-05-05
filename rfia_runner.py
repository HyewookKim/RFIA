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
    parser.add_argument('--data-root', dest='data_root', type=str, default='/root/Image_set', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args

def modify_representation_feature(representation_item,item,total_number):
    
    convert_to_total = representation_item[0]*total_number
    representation_feature = (convert_to_total + item[0])/(total_number+1)

    return representation_feature

def get_attention_feature(representation_item,cache):
    representation_feature = representation_item[0]
    cache_feature = [item[0] for item in cache]
    cache_loss = [item[1] for item in cache]
    feature_stack = torch.cat(cache_feature)
    gamma = 0.75
    loss_stack = torch.exp(-gamma*torch.cat(cache_loss))
    attention_score = F.softmax(torch.cosine_similarity(feature_stack,representation_feature,dim=1),dim=0)
    aligned_feature = torch.sum(attention_score.unsqueeze(1)*loss_stack.unsqueeze(1)*feature_stack,dim=0)
    aligned_feature /= aligned_feature.norm(dim=-1, keepdim=True)
    aligned_feature = aligned_feature.unsqueeze(0)
    aligned_item = [copy.deepcopy(representation_item)]
    aligned_item[0][0] = aligned_feature
    return aligned_item

def compare_representation_feature(old_representation_feature, new_representation_feature, pred, clip_weights) :

    text_embedding_of_pred = clip_weights[:, pred].unsqueeze(1)
    old_feature_score = 100*old_representation_feature @ text_embedding_of_pred
    new_feature_score = 100*new_representation_feature @ text_embedding_of_pred

    score = torch.cat((old_feature_score,new_feature_score),dim=1)
    score = F.softmax(score,dim=1).squeeze(0)

    if old_feature_score.item() < new_feature_score.item() :
        old_representation_feature = (old_representation_feature*score[0] + copy.deepcopy(new_representation_feature)*score[1])

    return old_representation_feature

def update_cache(cache, pred, features_loss, shot_capacity, num_count, clip_weights,representation_cache,attention_cache, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity: 
                cache[pred].append(item)
                num_count[pred] = num_count[pred] + 1
                total_number = num_count[pred]
                representation_item = representation_cache[pred][0]
                representation_item[0] = modify_representation_feature(representation_item,item,total_number)
                attention_cache[pred] = get_attention_feature(representation_item,cache[pred])
                representation_item[0] = compare_representation_feature(representation_item[0], attention_cache[pred][0][0], pred, clip_weights)
                            
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
                num_count[pred] = num_count[pred] + 1
                total_number = num_count[pred]
                representation_item = representation_cache[pred][0]
                representation_item[0] = modify_representation_feature(representation_item,item,total_number)
                attention_cache[pred] = get_attention_feature(representation_item,cache[pred])
                representation_item[0] = compare_representation_feature(representation_item[0], attention_cache[pred][0][0], pred, clip_weights)
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
            num_count[pred] = 0
            representation_item = copy.deepcopy(item)
            representation_item[1] = torch.zeros_like(item[1]) # item[1] is entropy. However, representation_item do not use entropy. So We set loss value = 0 for convenience
            representation_cache[pred] = [representation_item]

def merge_dictionary(dict1,dict2) :

    merged_dict = {}

    for key in set(dict1)|set(dict2) :

        if key in dict1 and key in dict2:
            merged_dict[key] = dict1[key][:] + [dict2[key][0]]

        elif key in dict1:
            merged_dict[key] = dict1[key][:]

        else:
            merged_dict[key] = dict2[key][:]

    return merged_dict
    
def compute_cache_logits(image_features, cache, representation_cache, attention_cache, alpha, beta, clip_weights):
    """Compute logits using positive/negative cache.""" 
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        merged_cache = merge_dictionary(cache, attention_cache)           
        for class_index in sorted(cache.keys()):
            merged_cache[class_index] = [representation_cache[class_index][0]] + [representation_cache[class_index][0]] + merged_cache[class_index][:]
            for item in merged_cache[class_index]:
                cache_keys.append(item[0])
                cache_values.append(class_index)                  
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits   

def run_test_tda(pos_cfg, loader, clip_model, clip_weights, wandb_test):
    with torch.no_grad():
        pos_cache, accuracies = {}, []
        representation_cache = {}
        attention_cache = {}
        pos_num_count = {}

        #Unpack all hyperparameters
        pos_enabled = pos_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}

        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
            target = target.cuda()
            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss, prob_map], pos_params['shot_capacity'], pos_num_count, clip_weights,
                                 representation_cache, attention_cache,True)
            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache,representation_cache,attention_cache, pos_params['alpha'], pos_params['beta'], clip_weights)

            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            if wandb_test :
                wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)
            if i%1000==0:
                print("---- RFIA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        print("---- RFIA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        return sum(accuracies)/len(accuracies)    



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)
    args.wandb = False
    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="RFIA", config=cfg, group=group_name, name=run_name)

        acc = run_test_tda(cfg['positive'], test_loader, clip_model, clip_weights, args.wandb)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()
if __name__ == "__main__":
    main()