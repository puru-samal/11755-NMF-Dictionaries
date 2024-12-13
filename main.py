import yaml
import torch
import os
import argparse
from typing import Literal
from dataset import MUSDBDataset
from torch.utils.data import DataLoader
from utils import batch_encode, batch_decode
from nmf import train_nmf_dictionary, test_separation, test_separation_semi
from datetime import datetime

def main(config_path, mode:Literal['train', 'test', 'test_semi']='train', B_target_pth=None, B_background_pth=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DEBUG = False   # Set to True to print debug information

    # create results directory
    id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f'experiments/{mode}/{id}'
    os.makedirs(path, exist_ok=True)

    ## Load config ---------------------------------------------------------------
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    ## --------------------------------------------------------------------------

    ## Load dataset -------------------------------------------------------------
    train_dataset = MUSDBDataset(
        root        = config['dataset']['data_dir'], 
        subset      = 'train', 
        target      = config['dataset']['target'], 
        sample_rate = config['dataset']['sample_rate'], 
    )
    print('Train dataset initialized\n')

    test_dataset = MUSDBDataset(
        root        = config['dataset']['data_dir'], 
        subset      = 'test', 
        target      = config['dataset']['target'], 
        sample_rate = config['dataset']['sample_rate'], 
    )
    print('Test dataset initialized\n')
    ## --------------------------------------------------------------------------

    ## Get item ------------------------------------------------------------------
    if DEBUG:   
        print("Train dataset: Getting item...")
        mixture, target, background = train_dataset[0]
        print(f"mixture.shape: {mixture.shape}, target.shape: {target.shape}, background.shape: {background.shape}\n")

        print("Test dataset: Getting item...")
        mixture, target, background = test_dataset[0]
        print(f"mixture.shape: {mixture.shape}, target.shape: {target.shape}, background.shape: {background.shape}\n")
    ## --------------------------------------------------------------------------

    ## DataLoader ----------------------------------------------------------------
    train_dataloader = DataLoader(
        dataset     = train_dataset, 
        batch_size  = config['dataloader']['batch_size'], 
        shuffle     = config['dataloader']['shuffle'], 
        num_workers = config['dataloader']['num_workers'],
        collate_fn  = lambda x: batch_encode(
                x, 
                n_fft       = config['transform']['n_fft'], 
                hop_length  = config['transform']['hop_length'], 
                center      = config['transform']['center'], 
                pad_mode    = config['transform']['pad_mode'],
            ),
    )


    test_dataloader = DataLoader(
        dataset     = test_dataset, 
        batch_size  = config['dataloader']['batch_size'],   
        shuffle     = config['dataloader']['shuffle'], 
        num_workers = config['dataloader']['num_workers'],
        collate_fn  = lambda x: batch_encode(
                x, 
                n_fft       = config['transform']['n_fft'], 
                hop_length  = config['transform']['hop_length'], 
                center      = config['transform']['center'], 
                pad_mode    = config['transform']['pad_mode'],
            ),
    )
    ## --------------------------------------------------------------------------

    ## Get batch -----------------------------------------------------------------
    if DEBUG:
        print("Train dataloader: Getting batch...")
        batch_mixture, batch_target, batch_background = next(iter(train_dataloader))
        print(f"batch_mixture.shape: {batch_mixture.shape}, batch_target.shape: {batch_target.shape}, batch_background.shape: {batch_background.shape}\n")

        print("Test dataloader: Getting batch...")
        batch_mixture, batch_target, batch_background = next(iter(test_dataloader))
        print(f"batch_mixture.shape: {batch_mixture.shape}, batch_target.shape: {batch_target.shape}, batch_background.shape: {batch_background.shape}\n")    
    ## --------------------------------------------------------------------------   

    ## Train Target NMF dictionary ----------------------------------------------
    if mode == 'train':
        print("Training target NMF dictionary...")
        B_target = train_nmf_dictionary(
            dataloader  = train_dataloader,
            device      = device,
            train_target= 'target',
            K           = config['nmf']['target_K'], 
            n_iter      = config['nmf']['n_iter'], 
            tol         = config['nmf']['tol'], 
            beta        = config['nmf']['beta'],
            alpha       = config['nmf']['alpha'],
        )
        print("Target NMF dictionary trained\n")
    ## Save dictionary ----------------------------------------------------------
        torch.save(B_target, f'{path}/{config["dataset"]["target"]}_{config["nmf"]["target_K"]}_nmf_dictionary.pth')
    ## --------------------------------------------------------------------------

    ## Train Background NMF dictionary -------------------------------------------
        print("Training background NMF dictionary...")
        B_background = train_nmf_dictionary(
            dataloader  = train_dataloader,
            device      = device,
            train_target= 'background',
            K           = config['nmf']['background_K'], 
            n_iter      = config['nmf']['n_iter'], 
            tol         = config['nmf']['tol'], 
            beta        = config['nmf']['beta'],
            alpha       = config['nmf']['alpha'],
        )
        print("Background NMF dictionary trained\n")
    ## Save dictionary ----------------------------------------------------------
        torch.save(B_background, f'{path}/background_{config["nmf"]["background_K"]}_nmf_dictionary.pth')
    ## --------------------------------------------------------------------------  

    ## Save Config --------------------------------------------------------------
    if mode == 'train':
        with open(f'{path}/{config["dataset"]["target"]}_config.yaml', 'w') as file:
            yaml.dump(config, file)
    ## --------------------------------------------------------------------------

    ## Test separation ----------------------------------------------------------
    if mode == 'test':
        if B_target_pth is None and B_background_pth is None:
            raise ValueError("B_target and B_background must be provided"   )
        print("Testing separation...")
        B_target = torch.load(B_target_pth, weights_only=True)
        B_background = torch.load(B_background_pth, weights_only=True)

        scores = test_separation(
            dataloader  = test_dataloader,
            B_target    = B_target,
            B_background= B_background,
            decode_fn   = lambda x: batch_decode(
                x, 
                n_fft=config['transform']['n_fft'], 
                hop_length=config['transform']['hop_length'], 
                center=config['transform']['center'],
            ),
            device      = device,
            n_iter      = config['nmf']['n_iter'],
            beta        = config['nmf']['beta'],
            alpha       = config['nmf']['alpha'],
            results_dir = path,
        )

        print("Separation tested\n")
        # print mean scores of each metric
        for metric in scores.keys():
            print(f"{metric}: {torch.tensor(scores[metric]).mean().item()}")
    ## --------------------------------------------------------------------------   

    ## Test semi-supervised separation ------------------------------------------
    if mode == 'test_semi':
        if B_target_pth is None:
            raise ValueError("B_target must be provided")
        print("Testing semi-supervised separation...")
        B_target = torch.load(B_target_pth, weights_only=True)
        scores = test_separation_semi(
            dataloader  = test_dataloader,
            B_target    = B_target,
            background_k= config['nmf']['background_K_semi'],
            decode_fn   = lambda x: batch_decode(x, n_fft=config['transform']['n_fft'], hop_length=config['transform']['hop_length'], center=config['transform']['center']),
            device      = device,
            n_iter      = config['nmf']['n_iter'],
            beta        = config['nmf']['beta'],
            alpha       = config['nmf']['alpha'],
            results_dir = path,
        )
        
        print("Separation tested\n")
        # print mean scores of each metric
        for metric in scores.keys():
            print(f"{metric}: {torch.tensor(scores[metric]).mean().item()}") 
    ## --------------------------------------------------------------------------

    ## Save scores --------------------------------------------------------------
    if mode == 'test' or mode == 'test_semi':  
        # conveert each np.array in scores to float
        for metric in scores.keys():
            scores[metric] = [round(float(item), 2) for item in scores[metric]]

        # save scores
        with open(f'{path}/scores.yaml', 'w') as file:
            yaml.dump(scores, file)
    ## --------------------------------------------------------------------------   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NMF source separation with config file')
    parser.add_argument('config_path', type=str, help='Path to config.yaml file')
    parser.add_argument('--mode', type=str, help='Mode: train or test or test_semi', default='train')
    parser.add_argument('--B_target_pth', type=str, help='Path to target NMF dictionary', default=None) 
    parser.add_argument('--B_background_pth', type=str, help='Path to background NMF dictionary', default=None)
    args = parser.parse_args()  
    main(args.config_path, args.mode, args.B_target_pth, args.B_background_pth)