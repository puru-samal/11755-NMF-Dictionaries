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


def train_nmf_dictionary(dataloader:torch.utils.data.DataLoader, K:int, train_target:Literal['target', 'background']='target', device:str='cpu', n_iter:int=100, tol:float=1e-4, beta:float=2.0, alpha:float=0.0):
    '''
    Train the NMF dictionary using custom implementation with CUDA optimization.
    '''
    # Get dimensions from first batch
    first_batch = next(iter(dataloader))[0]
    F = first_batch.shape[1]
    _B = torch.zeros(len(dataloader), F, K, device=device, dtype=torch.float32)
    
    # Move loss functions to device
    loss_fn = BetaDivergenceLoss(beta=beta).to(device)
    recons_loss_fn = MSELoss().to(device)

    # Pre-allocate tensors on device
    for i, (mixture, target, background) in enumerate(dataloader):
        print(f"Processing sample {i+1}/{len(dataloader)}")
        
        # Select target or background and clear previous iteration's cache
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        X = target if train_target == 'target' else background
        
        # Move to device and process in one go
        X = torch.abs(X.squeeze(0).to(device)) + 1e-12
        F, T = X.shape

        # Initialize B and W directly on device
        B = (torch.rand(F, K, device=device) + 1e-12)
        B /= torch.sum(B, dim=0, keepdim=True)
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
            with torch.cuda.amp.autocast(enabled=device=='cuda'):
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
        _B[i] = B.detach()
            
        if torch.isnan(_B[i]).any():
            print(f"\n\nWarning: NaN values detected in output at sample {i}\n\n")
            _B[i] = torch.zeros_like(_B[i])
        
        # Clear iteration variables
        del X, B, W, X_hat, loss, recons_loss
        if device == 'cuda':
            torch.cuda.empty_cache()
            
    # Final reshape and move to CPU before return
    result = _B.reshape(F, -1)
    if device == 'cuda':
        result = result.cpu()
        torch.cuda.empty_cache()
        
    return result


def test_separation(dataloader:torch.utils.data.DataLoader, B_target:torch.Tensor, B_background:torch.Tensor, decode_fn:Callable, device:str='cpu', n_iter:int=100, beta:float=2.0, alpha:float=0.0, results_dir:str=None):
    '''
    Test the separation of the NMF dictionary.
    '''
    os.makedirs(f'{results_dir}/audio', exist_ok=True)
    target_slice = slice(0, B_target.shape[-1])
    background_slice = slice(B_target.shape[-1], B_background.shape[-1])
    B_separation = torch.cat([B_target, B_background], dim=1).to(device)
    
    # Move loss functions to device
    loss_fn = BetaDivergenceLoss(beta=beta).to(device)
    recons_loss_fn = MSELoss().to(device)
    
    scores = {
        'SDR': [], 'ISR': [], 'SIR': [], 'SAR': [], 'Perm': []
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
            with torch.cuda.amp.autocast(enabled=device=='cuda'):
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
        scores['SDR'].append(sdr.mean())
        scores['ISR'].append(isr.mean())
        scores['SIR'].append(sir.mean())
        scores['SAR'].append(sar.mean())
        scores['Perm'].append(perm.mean())

        # Clear iteration variables
        del X, X_original, W, X_hat, target_mask, background_mask
        del predicted_target, predicted_background
        del target_audio, background_audio, predicted_target_audio, predicted_background_audio
        if device == 'cuda':
            torch.cuda.empty_cache()

    return scores




"""
def train_nmf_dictionary(dataloader:torch.utils.data.DataLoader, K:int, train_target:Literal['target', 'background']='target', device:str='cpu', n_iter:int=100, tol:float=1e-4, beta:float=1.0, alpha:float=0.0, l1_ratio:float=0.0):
    '''
    Train the NMF dictionary using torchnmf.
    '''
    first_batch = next(iter(dataloader))[0]
    B = torch.zeros(len(dataloader), first_batch.shape[1], K, device=device, dtype=torch.float32)
    
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
        model = NMF(Vshape=(X.shape), rank=K)
        
        # Ensure X is element-wise non-negative
        # count negative values
        negative_count = (X < 0).sum().item()
        if negative_count > 0:
            print(f"\n\nWarning: Negative values detected in input at batch {i}\n\n")
            print(f"Negative values: {negative_count}")
        
        # Check input for NaN
        if torch.isnan(X).any():
            print(f"\n\nWarning: NaN values detected in input at batch {i}\n\n")
            continue
            
        try:
            # Fit NMF model
            model.fit(X, beta=beta, tol=tol, max_iter=n_iter, verbose=True, alpha=alpha, l1_ratio=l1_ratio)
            H = model.H.detach().cpu()
            B[i] = H # Store the dictionary
            
            
        except Exception as e:
            print(f"\n\nError in NMF at batch {i}: {str(e)}\n\n")
            B[i] = torch.zeros_like(B[i])
            
        # Check output for NaN
        if torch.isnan(B[i]).any():
            print(f"\n\nWarning: NaN values detected in output at batch {i}\n\n")
            B[i] = torch.zeros_like(B[i])
            
    return B
"""