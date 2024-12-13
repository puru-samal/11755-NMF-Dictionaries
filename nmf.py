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
def update(V, B, W, beta=2.0, alpha=0.0):
    """
    Perform one iteration of NMF update
    Args:
        V: Target matrix (F x T)
        B: Basis matrix (F x K)
        W: Activation matrix (K x T)
        beta: Beta divergence parameter (default=2 for Euclidean distance)
    Returns:
        B_new, W_new: Updated matrices
    """
    
    # Update W
    BW = torch.matmul(B, W)
    numerator = B.T @ ((BW ** (beta-2)) * V)
    denominator = B.T @ (BW ** (beta-1))
    W *= numerator / (denominator + alpha)
    
    # Update B
    BW = torch.matmul(B, W)   # Recompute with new H
    numerator = ((BW ** (beta-2)) * V) @ W.T
    denominator = (BW ** (beta-1)) @ W.T
    B *= (numerator / denominator)

    # Normalize B to sum to 1 along frequency axis 
    bs = torch.sum(B, dim=0, keepdim=True)
    B /= bs
    W *= bs.T
    return B, W

def update_W(V, B, W, beta=2.0, alpha=0.0):
    '''
    Update the activation matrix W.
    Args:
        V: Target matrix (F x T)
        B: Basis matrix (F x K)
        W: Activation matrix (K x T)
        beta: Beta divergence parameter (default=2 for Euclidean distance)
    Returns:
        W_new: Updated activation matrix
    '''
    BW = torch.matmul(B, W)
    numerator = B.T @ ((BW ** (beta-2)) * V)
    denominator = B.T @ (BW ** (beta-1))
    W *= numerator / (denominator + alpha)
    return W    

def update_W_semi(V, B, Bn, W, Wn, alpha=0.0):
    '''
    Update the a portion of the activation matrix W in the background.
    '''
    V_BW = V / torch.matmul(B, W)
    numerator = B.T @ V_BW
    denominator = (B.T @ torch.ones_like(V_BW))
    W *= numerator / (denominator + alpha)
    return W

def update_B_semi(V, B, Bn, W, Wn, alpha=0.0):
    '''
    Update just the background basis matrix B.
    '''
    V_BW = V / torch.matmul(B, W)
    numerator = (V_BW @ Wn.T)
    denominator = torch.ones_like(V_BW) @ Wn.T
    Bn *= (numerator / denominator)
    return Bn

def train_nmf_dictionary(dataloader:torch.utils.data.DataLoader, K:int, train_target:Literal['target', 'background']='target', device:str='cpu', n_iter:int=100, tol:float=1e-4, beta:float=2.0, alpha:float=0.0):
    '''
    Train the NMF dictionary using custom implementation with CUDA optimization.
    '''
    # Get dimensions from first batch
    first_batch = next(iter(dataloader))[0]
    F = first_batch.shape[1]
    #_B = torch.zeros(len(dataloader), F, K, device=device, dtype=torch.float32)
    
    # Move loss functions to device
    loss_fn = BetaDivergenceLoss(beta=beta).to(device)
    recons_loss_fn = MSELoss().to(device)

    # Pre-allocate tensors on device
    # Initialize B and W directly on device
    B = (torch.rand(F, K, device=device) + 1e-12)
    B /= torch.sum(B, dim=0, keepdim=True)
    
    for i, (mixture, target, background) in enumerate(dataloader):
        print(f"Processing sample {i+1}/{len(dataloader)}")
        
        # Select target or background and clear previous iteration's cache
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        X = target if train_target == 'target' else background
        
        # Move to device and process in one go
        X = torch.abs(X.squeeze(0).to(device)) + 1e-12
        F, T = X.shape

        # Initialize W on device
        W = (torch.rand(K, T, device=device) + 1e-12)
        
        if torch.isnan(X).any():
            print(f"\n\nWarning: NaN values detected in input at sample {i}\n\n")
            continue
            
        # Iterative update with in-place operations where possible
        prev_loss = float('inf')
        pbar = tqdm(range(n_iter), desc=f'NMF iterations (sample {i+1})', leave=False)
        
        for _ in pbar:
            # Update B and W in-place on GPU
            B, W = update(X, B, W, beta=beta, alpha=alpha)
            
            # Compute losses without transferring to CPU
            X_hat = torch.matmul(B, W)
            loss = loss_fn(X, X_hat)
            recons_loss = recons_loss_fn(X, X_hat)
            
            pbar.set_postfix({'beta_loss': f'{loss.item():.4f}', 'recons_loss': f'{recons_loss.item():.4f}'})
            
            # Check convergence using GPU values
            if abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()
        
        print(f"Beta divergence loss: {loss.item():.4f}, Reconstruction loss: {recons_loss.item():.4f}")
        
        # Store result directly on device
        #_B[i] = B.detach()
            
        #if torch.isnan(_B[i]).any():
        #    print(f"\n\nWarning: NaN values detected in output at sample {i}\n\n")
        #    _B[i] = torch.zeros_like(_B[i])
        
        # Clear iteration variables
        del X, W, X_hat, loss, recons_loss
        if device == 'cuda':
            torch.cuda.empty_cache()
            
    # Final reshape and move to CPU before return
    #result = _B.reshape(F, -1)
    if device == 'cuda':
        #result = result.cpu()
        torch.cuda.empty_cache()
        
    return B.detach().cpu()

def test_separation_semi(dataloader:torch.utils.data.DataLoader, B_target:torch.Tensor, background_k:int, decode_fn:Callable, device:str='cpu', n_iter:int=100, beta:float=2.0, alpha:float=0.0, results_dir:str=None):
    '''
    Test the separation of the NMF dictionary.
    '''
    os.makedirs(f'{results_dir}/audio', exist_ok=True)
    target_slice = slice(0, B_target.shape[-1])
    B_background = torch.abs(torch.rand(B_target.shape[0], background_k, device=device))
    background_slice = slice(B_target.shape[-1], (B_target.shape[-1]+B_background.shape[-1]))
    B_separation = torch.cat([B_target, B_background], dim=1).to(device)
    
    # Move loss functions to device
    loss_fn = BetaDivergenceLoss(beta=beta).to(device)
    recons_loss_fn = MSELoss().to(device)
    
    scores = {
        'SDR_target': [], 'ISR_target': [], 'SIR_target': [], 'SAR_target': [], 'Perm_target': [],
        'SDR_background': [], 'ISR_background': [], 'SIR_background': [], 'SAR_background': [], 'Perm_background': []
    }

    for i, (mixture, target, background) in enumerate(dataloader):
        print(f"Processing sample {i+1}/{len(dataloader)}")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Move data to device and process
        X = mixture.squeeze(0).to(device)
        X_original = X.clone()
        X = torch.abs(X) + 1e-12

        # Initialize W on device
        W = torch.abs(torch.rand(B_separation.shape[1], X.shape[1], device=device) + 1e-12)

        if torch.isnan(X).any():
            print(f"\n\nWarning: NaN values detected in input at sample {i}\n\n")
            continue    
        
        # Iterative update with in-place operations
        prev_loss = float('inf')
        pbar = tqdm(range(n_iter), desc=f'NMF iterations (sample {i+1})', leave=False)
        
        for _ in pbar:  
            # Update W in-place on device
            W = update_W_semi(X, B_separation, B_separation[:, background_slice], W, W[background_slice, :], alpha=alpha)
            B_separation[:, background_slice] = update_B_semi(X, B_separation, B_separation[:, background_slice], W, W[background_slice, :], alpha=alpha)

            # Compute losses without transferring to CPU
            X_hat = torch.matmul(B_separation, W)
            loss = loss_fn(X, X_hat)
            recons_loss = recons_loss_fn(X, X_hat)
            
            pbar.set_postfix({'beta_loss': f'{loss.item():.4f}', 'recons_loss': f'{recons_loss.item():.4f}'})

        print(f"Beta divergence loss: {loss.item():.4f}, Reconstruction loss: {recons_loss.item():.4f}")
        
        # Compute masks and predictions on device
        BW_sum = (B_separation[:, target_slice] @ W[target_slice, :]) + (B_separation[:, background_slice] @ W[background_slice, :])
        target_mask = (B_separation[:, target_slice] @ W[target_slice, :]) / BW_sum
        background_mask = (B_separation[:, background_slice] @ W[background_slice, :]) / BW_sum
        predicted_target = X_original * target_mask
        predicted_background = X_original * background_mask

        # Move to CPU for audio processing
        target_audio, background_audio = decode_fn((target.squeeze(0), background.squeeze(0)))
        predicted_target_audio, predicted_background_audio = decode_fn((predicted_target.cpu(), predicted_background.cpu()))

        # Save audio
        torchaudio.save(f'{results_dir}/audio/predicted_target_{i}.wav', predicted_target_audio.unsqueeze(0), 16000, channels_first=True)
        torchaudio.save(f'{results_dir}/audio/predicted_background_{i}.wav', predicted_background_audio.unsqueeze(0), 16000, channels_first=True)

        # Evaluate on CPU
        reference = torch.stack([target_audio.reshape(-1,1), background_audio.reshape(-1,1)]).numpy()
        prediction = torch.stack([predicted_target_audio.reshape(-1,1), predicted_background_audio.reshape(-1,1)]).numpy()
        print(f"reference.shape: {reference.shape}, prediction.shape: {prediction.shape}")
        
        sdr, isr, sir, sar, perm = museval.metrics.bss_eval(reference, prediction)
        scores['SDR_target'].append(np.nanmean(sdr, axis=1)[0])
        scores['SDR_background'].append(np.nanmean(sdr, axis=1)[1])
        scores['ISR_target'].append(np.nanmean(isr, axis=1)[0])
        scores['ISR_background'].append(np.nanmean(isr, axis=1)[1])
        scores['SIR_target'].append(np.nanmean(sir, axis=1)[0])
        scores['SIR_background'].append(np.nanmean(sir, axis=1)[1])
        scores['SAR_target'].append(np.nanmean(sar, axis=1)[0])
        scores['SAR_background'].append(np.nanmean(sar, axis=1)[1])
        scores['Perm_target'].append(np.nanmean(perm, axis=1)[0])
        scores['Perm_background'].append(np.nanmean(perm, axis=1)[1])

        print(f"SDR_target: {round(float(scores['SDR_target'][-1]), 2)}, ISR_target: {round(float(scores['ISR_target'][-1]), 2)}, SIR_target: {round(float(scores['SIR_target'][-1]), 2)}, SAR_target: {round(float(scores['SAR_target'][-1]), 2)}, Perm_target: {round(float(scores['Perm_target'][-1]), 2)}")
        print(f"SDR_background: {round(float(scores['SDR_background'][-1]), 2)}, ISR_background: {round(float(scores['ISR_background'][-1]), 2)}, SIR_background: {round(float(scores['SIR_background'][-1]), 2)}, SAR_background: {round(float(scores['SAR_background'][-1]), 2)}, Perm_background: {round(float(scores['Perm_background'][-1]), 2)}\n\n")
        
        # Clear iteration variables
        del X, X_original, W, X_hat, target_mask, background_mask
        del predicted_target, predicted_background
        del target_audio, background_audio, predicted_target_audio, predicted_background_audio
        if device == 'cuda':
            torch.cuda.empty_cache()

    return scores

def test_separation(dataloader:torch.utils.data.DataLoader, B_target:torch.Tensor, B_background:torch.Tensor, decode_fn:Callable, device:str='cpu', n_iter:int=100, beta:float=2.0, alpha:float=0.0, results_dir:str=None):
    '''
    Test the separation of the NMF dictionary.
    '''
    os.makedirs(f'{results_dir}/audio', exist_ok=True)
    target_slice = slice(0, B_target.shape[-1])
    background_slice = slice(B_target.shape[-1], (B_target.shape[-1]+B_background.shape[-1]))
    B_separation = torch.cat([B_target, B_background], dim=1).to(device)
    
    # Move loss functions to device
    loss_fn = BetaDivergenceLoss(beta=beta).to(device)
    recons_loss_fn = MSELoss().to(device)
    
    scores = {
        'SDR_target': [], 'ISR_target': [], 'SIR_target': [], 'SAR_target': [], 'Perm_target': [],
        'SDR_background': [], 'ISR_background': [], 'SIR_background': [], 'SAR_background': [], 'Perm_background': []
    }
    
    for i, (mixture, target, background) in enumerate(dataloader):
        print(f"Processing sample {i+1}/{len(dataloader)}")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Move data to device and process
        X = mixture.squeeze(0).to(device)
        X_original = X.clone()
        X = torch.abs(X) + 1e-12

        # Initialize W on device
        W = torch.abs(torch.rand(B_separation.shape[1], X.shape[1], device=device) + 1e-12)

        if torch.isnan(X).any():
            print(f"\n\nWarning: NaN values detected in input at sample {i}\n\n")
            continue    
        
        # Iterative update with in-place operations
        prev_loss = float('inf')
        pbar = tqdm(range(n_iter), desc=f'NMF iterations (sample {i+1})', leave=False)
        
        for _ in pbar:  
            # Update W in-place on device
            W = update_W(X, B_separation, W, beta=beta, alpha=alpha)

            # Compute losses without transferring to CPU
            X_hat = torch.matmul(B_separation, W)
            loss = loss_fn(X, X_hat)
            recons_loss = recons_loss_fn(X, X_hat)
            
            pbar.set_postfix({'beta_loss': f'{loss.item():.4f}', 'recons_loss': f'{recons_loss.item():.4f}'})

        print(f"Beta divergence loss: {loss.item():.4f}, Reconstruction loss: {recons_loss.item():.4f}")
        
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

        # Evaluate on CPU
        reference = torch.stack([target_audio.reshape(-1,1), background_audio.reshape(-1,1)]).numpy()
        prediction = torch.stack([predicted_target_audio.reshape(-1,1), predicted_background_audio.reshape(-1,1)]).numpy()
        print(f"reference.shape: {reference.shape}, prediction.shape: {prediction.shape}")

        
        sdr, isr, sir, sar, perm = museval.metrics.bss_eval(reference, prediction)

        scores['SDR_target'].append(np.nanmean(sdr, axis=1)[0])
        scores['SDR_background'].append(np.nanmean(sdr, axis=1)[1])
        scores['ISR_target'].append(np.nanmean(isr, axis=1)[0])
        scores['ISR_background'].append(np.nanmean(isr, axis=1)[1])
        scores['SIR_target'].append(np.nanmean(sir, axis=1)[0])
        scores['SIR_background'].append(np.nanmean(sir, axis=1)[1])
        scores['SAR_target'].append(np.nanmean(sar, axis=1)[0])
        scores['SAR_background'].append(np.nanmean(sar, axis=1)[1])
        scores['Perm_target'].append(np.nanmean(perm, axis=1)[0])
        scores['Perm_background'].append(np.nanmean(perm, axis=1)[1])

        print(f"SDR_target: {round(float(scores['SDR_target'][-1]), 2)}, ISR_target: {round(float(scores['ISR_target'][-1]), 2)}, SIR_target: {round(float(scores['SIR_target'][-1]), 2)}, SAR_target: {round(float(scores['SAR_target'][-1]), 2)}, Perm_target: {round(float(scores['Perm_target'][-1]), 2)}")
        print(f"SDR_background: {round(float(scores['SDR_background'][-1]), 2)}, ISR_background: {round(float(scores['ISR_background'][-1]), 2)}, SIR_background: {round(float(scores['SIR_background'][-1]), 2)}, SAR_background: {round(float(scores['SAR_background'][-1]), 2)}, Perm_background: {round(float(scores['Perm_background'][-1]), 2)}\n\n")
        
        # Clear iteration variables
        del X, X_original, W, X_hat, target_mask, background_mask
        del predicted_target, predicted_background
        del target_audio, background_audio, predicted_target_audio, predicted_background_audio
        if device == 'cuda':
            torch.cuda.empty_cache()

    return scores
