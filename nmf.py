import torch
from torchnmf.nmf import NMF
from tqdm import tqdm
from typing import Literal
import matplotlib.pyplot as plt
from losses import BetaDivergenceLoss
from torch.nn import MSELoss
import museval
from typing import Callable
import torchaudio
import os
import numpy as np

def train_nmf_dictionary(
        dataloader:torch.utils.data.DataLoader, 
        K:int, 
        train_target:Literal['target', 'background']='target', 
        device:str='cpu', 
        n_iter:int=100, 
        tol:float=1e-4, 
        beta:float=1.0, 
        alpha:float=0.0, 
        l1_ratio:float=0.0,
        num_samples:int=10
    ):
    '''
    Train the NMF dictionary using torchnmf.
    Args:
        dataloader: torch.utils.data.DataLoader,
        K: int,
        train_target: Literal['target', 'background'],
        device: str,
        n_iter: int,
        tol: float,
        beta: float,
        alpha: float,
        l1_ratio: float
    '''
    first_batch = next(iter(dataloader))[0]
    F = first_batch.shape[1]
    B = torch.zeros(num_samples, F, K, device=device, dtype=torch.float32)
    
    for i, (mixture, target, background) in enumerate(dataloader):
        print(f"Processing sample {i+1}/{len(dataloader)}")
        if train_target == 'target':
            X = target
        else:
            X = background

        # Take magnitude of the stft
        X = X.squeeze(0)
        X = torch.abs(X) + 1e-12

        # Initialize NMF model
        model = NMF(Vshape=(X.shape), rank=K).to(device)
        # Fit NMF model
        model.fit(X.to(device), beta=beta, tol=tol, max_iter=n_iter, verbose=True, alpha=alpha, l1_ratio=l1_ratio)
        B[i] = model.H.detach().cpu() # Store the dictionary
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

        if i == num_samples - 1:
            break

    # Final reshape and move to CPU before return
    result = B.reshape(F, -1)
    if device == 'cuda':
        result = result.cpu()
        torch.cuda.empty_cache()
        
    return result


def test_separation(
        dataloader:torch.utils.data.DataLoader, 
        B_target:torch.Tensor, 
        B_background:torch.Tensor, 
        decode_fn:Callable, 
        device:str='cpu', 
        n_iter:int=100, 
        beta:float=2.0, 
        alpha:float=0.0, 
        l1_ratio:float=0.0, 
        results_dir:str=None
    ):
    '''
    Test the separation of the NMF dictionary.
    '''
    os.makedirs(f'{results_dir}/audio', exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)
    target_slice = slice(0, B_target.shape[-1])
    background_slice = slice(B_target.shape[-1], B_background.shape[-1])
    B_separation = torch.cat([B_target, B_background], dim=1).to(device)

    scores = {}
    
    for i, (mixture, target, background) in enumerate(dataloader):
        print(f"Processing sample {i+1}/{len(dataloader)}")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Move data to device and process
        X = mixture.squeeze(0).to(device)
        X_original = X.clone()
        X = torch.abs(X) + 1e-12

        # Initialize NMF model
        model = NMF(Vshape=(X.shape), rank=B_separation.shape[1], H=B_separation, trainable_H=False).to(device)
        # Fit NMF model
        model.fit(X.to(device), beta=beta, max_iter=n_iter, verbose=True, alpha=alpha, l1_ratio=l1_ratio)
        W = model.W.T
        # Compute masks and predictions on device
        X_hat = B_separation @ W
        target_mask = (B_separation[:, target_slice] @ W[target_slice, :]) / X_hat
        background_mask = (B_separation[:, background_slice] @ W[background_slice, :]) / X_hat
        predicted_target = X_original * target_mask
        predicted_background = X_original * background_mask

        # Move to CPU for audio processing
        target_audio, background_audio = decode_fn((target.squeeze(0), background.squeeze(0)))
        predicted_target_audio, predicted_background_audio = decode_fn((predicted_target.cpu(), predicted_background.cpu()))

        # Save audio
        torchaudio.save(f'{results_dir}/audio/predicted_target_{i}.wav', predicted_target_audio.unsqueeze(0), 16000, channels_first=True)
        torchaudio.save(f'{results_dir}/audio/predicted_background_{i}.wav', predicted_background_audio.unsqueeze(0), 16000, channels_first=True)

        # Save target and background masks
        plt.imsave(f'{results_dir}/plots/target_mask_{i}.png', torch.abs(predicted_target).detach().cpu().numpy(), cmap='viridis')
        plt.imsave(f'{results_dir}/plots/background_mask_{i}.png', torch.abs(predicted_background).detach().cpu().numpy(), cmap='viridis')

        # Evaluate on CPU
        reference = torch.stack([target_audio.reshape(-1,1), background_audio.reshape(-1,1)]).detach().cpu().numpy()
        prediction = torch.stack([predicted_target_audio.reshape(-1,1), predicted_background_audio.reshape(-1,1)]).detach().cpu().numpy()

        
        sdr, isr, sir, sar, perm = museval.metrics.bss_eval(reference, prediction)
        # Remove nans   
        sdr = np.nan_to_num(sdr)
        isr = np.nan_to_num(isr)
        sir = np.nan_to_num(sir)
        sar = np.nan_to_num(sar)
        perm = np.nan_to_num(perm)
        scores[i] = {
            'SDR': {
                'target': round(float(sdr.mean(axis=1)[0]), 2),
                'background': round(float(sdr.mean(axis=1)[1]), 2)
            },
            'ISR': {
                'target': round(float(isr.mean(axis=1)[0]), 2),
                'background': round(float(isr.mean(axis=1)[1]), 2)
            },
            'SIR': {
                'target': round(float(sir.mean(axis=1)[0]), 2),
                'background': round(float(sir.mean(axis=1)[1]), 2)
            },
            'SAR': {
                'target': round(float(sar.mean(axis=1)[0]), 2),
                'background': round(float(sar.mean(axis=1)[1]), 2)
            },
            'Perm': {
                'target': round(float(perm.mean(axis=1)[0]), 2),
                'background': round(float(perm.mean(axis=1)[1]), 2)
            }
        }

        # Clear iteration variables
        del X, X_original, W, X_hat, target_mask, background_mask, model
        del predicted_target, predicted_background
        del target_audio, background_audio, predicted_target_audio, predicted_background_audio
        if device == 'cuda':
            torch.cuda.empty_cache()

    return scores