from utils import *


def combine_datasets(dataset_names = []):
    
    result = []
    for dataset in dataset_names:
        size += len(dataset)
        result += dataset
    
    gt_result = []


    label = 0
    for dataset in dataset_names:
        gt_result += [label]*len(dataset)
        label += 1

    return result, np.array(gt_result)
        



def main():
    if DOWNLOAD_DATA:
        download_datasets()
    

    #loading graphs
    fb = load_graph(min_num_nodes=100, name='facebook_ct1')
    github = load_graph(min_num_nodes=950, name='github_stargazers')
    reddit = load_graph(min_num_nodes=3200, name='REDDIT-BINARY')
    deezer = load_graph(min_num_nodes = 200, name = 'deezer_ego_nets')
    


    fb_github_reddit, gt_fb_github_reddit = combine_datasets([fb, github, reddit])
    # gt_fb_github_reddit


    fb_github_deezer, gt_fb_github_deezer = combine_datasets([fb, github, deezer])
    # gt_fb_github_deezer

    fb_reddit_deezer, gt_fb_reddit_deezer = combine_datasets([fb, reddit, deezer])
    # gt_fb_reddit_deezer


    github_reddit_deezer, gt_github_reddit_deezer = combine_datasets([github, reddit, deezer])
    # gt_github_reddit_deezer

    fb_github_reddit_deezer, gt_fb_github_reddit_deezer = combine_datasets([fb, github, reddit, deezer])
    # gt_fb_github_reddit_deezer