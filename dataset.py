import numpy as np
np.float_ = np.float32
import musdb
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
from torchaudio import transforms as T

class MUSDBDataset(Dataset):
    def __init__(self, root:str, subset:str, target:str, sample_rate:int):
        """
        Initialize the MUSDBDataset.

        Args:
            root   (str): The root directory of the MUSDB dataset.
            subset (str): The subset of the dataset to use (e.g., 'train', 'test', 'valid').
            target (str): The target instrument to use (e.g., 'vocals', 'drums', 'bass', 'other').
            sample_rate (int): The sample rate to resample the audio to.
        """
        
        self.mus = musdb.DB(root=root, subsets=subset)
        self.target = target
        self.sample_rate = sample_rate
        
        print(f"Initializing MUSDBDataset for {subset} subset, target: {target}")
        print(f"Processing tracks with target sample rate {sample_rate}")
        self.tracks = []
        for track in tqdm(self.mus, desc="Processing tracks", total=int(len(self.mus)), unit="track"):
            # Store track reference instead of loading audio
            self.tracks.append(track)
        print(f"Finished processing {len(self.tracks)} tracks.")

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        '''
        Get a single item from the dataset.
        Input: idx is the index of the item to get
        Output: a tuple (mixture, target)
        mixture: torch.Tensor of shape (time_steps)
        target: torch.Tensor of shape (time_steps)
        background: torch.Tensor of shape (time_steps)
        '''
        track = self.tracks[idx]
        
        # Get the audio data for the mixture and target
        mixture = torch.tensor(track.audio)
        target = torch.tensor(track.targets[self.target].audio)

        # Convert to mono
        mixture = torch.mean(mixture, dim=1)
        target  = torch.mean(target,  dim=1)
        
        # Resample if needed
        if self.sample_rate != 44100:
            resample = T.Resample(orig_freq=44100, new_freq=self.sample_rate, dtype=mixture.dtype)
            mixture = resample(mixture)
            target = resample(target)
          
        return mixture, target, mixture - target