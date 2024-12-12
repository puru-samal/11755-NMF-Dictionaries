import musdb
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_musdb_to_stft(
    musdb_root: str,
    output_dir: str,
    target: str = "vocals",
    subset: str = "train",
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann"
):
    """
    Process MUSDB tracks to STFT.
    
    Args:
        musdb_root (str): Path to MUSDB dataset
        output_dir (str): Where to save the STFTs
        target (str): Which stem to process ('vocals', 'drums', 'bass', 'other')
        subset (str): Which subset to process ('train' or 'test')
        sample_rate (int): Target sample rate
        n_fft (int): Size of FFT window
        hop_length (int): Number of samples between successive frames
        window (str): Window function to use
    """
    
    print(f"\nProcessing {subset} subset, target: {target}")
    
    # Initialize MUSDB
    mus = musdb.DB(root=musdb_root, subsets=subset)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create window
    window_tensor = torch.hann_window(n_fft) if window == "hann" else None
    
    # Create progress bars
    pbar_tracks = tqdm(mus, desc="Tracks", total=len(mus))
    
    # Process each track
    for track in pbar_tracks:
        track_name = track.name
        pbar_tracks.set_description(f"Processing track: {track_name}")
        
        # Get audio data
        mixture = torch.tensor(track.audio)
        target_audio = torch.tensor(track.targets[target].audio)
        
        # Convert to mono
        mixture = torch.mean(mixture, dim=1)
        target_audio = torch.mean(target_audio, dim=1)
        
        # Compute STFTs
        mixture_stft = torch.stft(
            mixture,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_tensor,
            return_complex=True
        )
        
        target_stft = torch.stft(
            target_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_tensor,
            return_complex=True
        )
        
        # Convert to numpy and separate real/imaginary parts
        mixture_stft_np = mixture_stft.numpy()
        target_stft_np = target_stft.numpy()
        
        # Save STFTs
        save_path = output_path / track_name
        save_path.mkdir(exist_ok=True)
        
        # Save as numpy arrays
        np.save(save_path / f"mixture_stft.npy", mixture_stft_np)
        np.save(save_path / f"{target}_stft.npy", target_stft_np)
        
        # Save metadata
        metadata = {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window": window,
            "shape_mixture": mixture_stft_np.shape,
            "shape_target": target_stft_np.shape
        }
        np.save(save_path / "metadata.npy", metadata)
    
    print(f"Finished processing {len(mus)} tracks for {subset} subset")

if __name__ == "__main__":
    # Example usage
    musdb_root = "path/to/musdb"
    output_dir = "path/to/output/stfts"
    
    # Process all subsets with a progress bar
    subsets = ["train", "test"]
    for subset in tqdm(subsets, desc="Subsets"):
        process_musdb_to_stft(
            musdb_root=musdb_root,
            output_dir=output_dir,
            target="vocals",
            subset=subset,
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            window="hann"
        )
    
    print("\nAll processing complete!")