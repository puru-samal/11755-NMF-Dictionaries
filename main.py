import yaml
import torch
import os
import argparse
from dataset import MUSDBDataset
from torch.utils.data import DataLoader
from utils import batch_encode, batch_decode
from nmf import train_nmf_dictionary, test_separation
from datetime import datetime

def main(config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DEBUG = False   # Set to True to print debug information

    # create results directory
    id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f'results/{id}', exist_ok=True)

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
        l1_ratio    = config['nmf']['l1_ratio'],
    )
    print("Target NMF dictionary trained\n")
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
        l1_ratio    = config['nmf']['l1_ratio'],
    )
    print("Background NMF dictionary trained\n")
    ## --------------------------------------------------------------------------   

    ## Save dictionary ----------------------------------------------------------
    # save to results directory
    torch.save(B_target, f'results/{id}/{config["dataset"]["target"]}_nmf_dictionary.pth')
    torch.save(B_background, f'results/{id}/background_nmf_dictionary.pth')
    ## --------------------------------------------------------------------------

    ## Save Config --------------------------------------------------------------
    with open(f'results/{id}/{config["dataset"]["target"]}_config.yaml', 'w') as file:
        yaml.dump(config, file)
    ## --------------------------------------------------------------------------

    ## Test separation ----------------------------------------------------------
    print("Testing separation...")
    scores = test_separation(
        dataloader  = test_dataloader,
        B_target    = B_target,
        B_background= B_background,
        decode_fn   = lambda x: batch_decode(x, n_fft=config['transform']['n_fft'], hop_length=config['transform']['hop_length'], center=config['transform']['center']),
        device      = device,
        n_iter      = config['nmf']['n_iter'],
        beta        = config['nmf']['beta'],
        alpha       = config['nmf']['alpha'],
        results_dir = f'results/{id}',
        l1_ratio    = config['nmf']['l1_ratio'],
    )
    print("Separation tested\n")
    # print scores
    for i, values in scores.items():
        print(f"{i}:")
        for metric, value in values.items():
            print(f"    {metric}: {value}")
        print("\n")

    ## --------------------------------------------------------------------------

    ## Print and Save scores --------------------------------------------------------------
    # dump to yaml file   
    with open(f'results/{id}/scores.yaml', 'w') as file:
        yaml.dump(scores, file)
    ## --------------------------------------------------------------------------   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NMF source separation with config file')
    parser.add_argument('config_path', type=str, help='Path to config.yaml file')
    args = parser.parse_args()
    main(args.config_path)