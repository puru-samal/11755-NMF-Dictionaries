import museval
import torch
from torchaudio import transforms as T
    
def batch_encode(batch, n_fft, hop_length, center, pad_mode):
    '''
    Input: batch is a tuple of 2D arrays of shape (n_samples, time_steps)
    Output: a tuple of 3D arrays of shape (n_samples, n_freq, n_time)
    ''' 
    # Unpack batch into separate lists
    mixture, target, background = zip(*batch)
    
    # Convert tuples to tensors
    mixture    = torch.stack(mixture)
    target     = torch.stack(target)
    background = torch.stack(background)
    
    # Take STFT
    mixture_stft = torch.stft(
        mixture, 
        n_fft       = n_fft, 
        hop_length  = hop_length, 
        return_complex=True,
        center=center,
        pad_mode=pad_mode,
        window=torch.hann_window(n_fft)
    )
    target_stft = torch.stft(
        target, 
        n_fft       = n_fft, 
        hop_length  = hop_length, 
        return_complex=True,
        center=center,
        pad_mode=pad_mode,
        window=torch.hann_window(n_fft)
    )

    background_stft = torch.stft(
        background,
        n_fft       = n_fft, 
        hop_length  = hop_length, 
        return_complex=True,
        center=center,
        pad_mode=pad_mode,
        window=torch.hann_window(n_fft)
    )

    return mixture_stft, target_stft, background_stft



def batch_decode(batch, n_fft, hop_length, center):
    '''
    Inverse of batch_encode.
    Input: batch is a tuple of 3D arrays of shape (n_samples, n_freq, n_time)
    Output: a tuple of 2D arrays of shape (n_samples, time_steps)
    '''
    pred_target, pred_background = batch

    # Inverse STFT
    pred_target = torch.istft(
        pred_target, 
        n_fft       = n_fft, 
        hop_length  = hop_length, 
        center      = center, 
        window=torch.hann_window(n_fft)
    )
    pred_background = torch.istft(
        pred_background, 
        n_fft       = n_fft, 
        hop_length  = hop_length, 
        center      = center, 
        window=torch.hann_window(n_fft)
    )
    
    return pred_target, pred_background


