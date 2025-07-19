#!/usr/bin/env python

import os, sys, gc, json, warnings
from pathlib import Path
from datetime import datetime

import torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────────────────────────────────────────
# 0 ▸ Path tweaks so imports work exactly like the notebook
# ───────────────────────────────────────────────────────────────
PARENT = "/home/jupyter/cnvae-ecg-us-central1-xdgfm/cNVAE_ECG-main"
sys.path.insert(0, PARENT)
sys.path.insert(1, f"{PARENT}/conditional")


from pathlib import Path
# ------------------------------------------------------------------
# All preprocessing cells saved their artefacts here ↓
# ------------------------------------------------------------------
DATA_ROOT  = Path("/home/jupyter/cnvae-ecg-us-central1-xdgfm")
OUTPUT_DIR = DATA_ROOT / "sleep_eda_output"        # <‑‑ fixed!
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────
# 1 ▸  >>>>>>>  COPY ‑‑ Cell 15 code goes **here**  <<<<<<
#     Everything that builds your Dataset + DataLoaders
#     At the end you must have:
#        ─ dataloaders["train"]  and  dataloaders["val"]
# ───────────────────────────────────────────────────────────────
# (Paste your entire notebook Cell 15 between the two triple quotes)

# ===============================================================
# 15 ▸  DATA LOADERS & AUGMENTATION FOR cNVAE‑ECG TRAINING
#       ↳ returns lead III / aVR / aVL / aVF as derived_leads
# ===============================================================
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy   as np
import pandas  as pd
import gc, psutil, time, warnings, json, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.preprocessing import RobustScaler
from scipy.signal import butter, filtfilt, resample
from typing import Dict
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("whitegrid")

# ===============================================================
# 15.1  SleepECGData
# ===============================================================
class SleepECGData(Dataset):

    # ---------- INIT ----------------------------------------------------------
    def __init__(self,
                 manifest_csv      : str,
                 split             : str = "train",
                 augment           : bool = True,
                 cache_size        : int  = 1_000,
                 config            : dict | None = None,
                 preload_clinical  : bool = True,
                 global_mean=None, global_std=None, **kw):
        super().__init__()
        self.split        = split
        self.augment      = augment and split == "train"
        self.cache_size   = cache_size
        self.signal_cache = {}
        self.cache_hits   = 0
        self.cache_misses = 0
        self.gμ           = global_mean
        self.gσ           = global_std

        # ---------- manifest --------------------------------------------------
        print(f"📂 Loading manifest for {split} split …")
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        print(f"   {len(self.manifest):,} windows")

        # ---------- config ----------------------------------------------------
        self.config = config or self._default_config()
        self._init_signal_processing()

        # ---------- clinical --------------------------------------------------
        self.clinical_data = None
        if preload_clinical:
            self._preload_clinical_data()

        # ---------- channels --------------------------------------------------
        self._init_channel_mapping()

        # ---------- augmentation ---------------------------------------------
        if self.augment:
            self._init_augmentation()

        print(f"✅ {split.upper()} dataset ready | "
              f"patients = {self.manifest['ParticipantKey'].nunique()} | "
              f"cache = {cache_size} | augment = {self.augment}")
    
    # ---------- helpers -------------------------------------------------------
    def _default_config(self):
        return dict(
            target_fs         = 128,
            window_length     = 15,
            normalize_method  = "robust",
            filter_ecg        = True,
            filter_psg        = True,
            ecg_freq_range    = (0.5, 40),
            psg_freq_range    = (0.1, 20),
            clinical_features = 5,
        )

    def _init_signal_processing(self):
        self.target_fs      = self.config["target_fs"]
        self.window_samples = int(self.config["window_length"] * self.target_fs)
        ny   = self.target_fs / 2
        if self.config["filter_ecg"]:
            lo, hi       = self.config["ecg_freq_range"]
            self.ecg_ba  = butter(4, [lo/ny, hi/ny], btype="band")
        if self.config["filter_psg"]:
            lo, hi       = self.config["psg_freq_range"]
            self.psg_ba  = butter(4, [lo/ny, hi/ny], btype="band")

    def _preload_clinical_data(self):
        print("📊 Pre‑loading clinical features …")
        cols = [c for c in self.manifest.columns
                if c in ["log_AHI","ptage","BMI","Sex_Female","Sex_Male"]]
        if not cols:
            print("   ⚠️  none found"); return
        df  = self.manifest[["file","win"]+cols].copy()
        num = [c for c in cols if df[c].dtype!="bool"]
        df[num] = df[num].fillna(df[num].median())
        df[cols] = RobustScaler().fit_transform(df[cols])
        self.clinical_data = df
        print(f"   {len(cols)} features loaded")

    def _init_channel_mapping(self):
        self.channel_names = ["ECG","Thor RIP","Abdo RIP","Airflow",
                              "CPAP Flow","Nasal Pressure","SpO₂","Position"]
        self.ecg_channel   = 0
        self.psg_channels  = list(range(1, 8))
        self.expected_ch   = len(self.channel_names)

    def _init_augmentation(self):
        self.aug_cfg = dict(
            time_shift_prob       = 0.3,  time_shift_max      = 0.10,
            amp_scale_prob        = 0.4,  amp_scale_range     = (0.8,1.2),
            noise_prob            = 0.2,  noise_level         = 0.02,
            channel_dropout_prob  = 0.1,  channel_dropout_max = 2,
        )

    # ---------- dataset protocol ---------------------------------------------
    def __len__(self): return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row       = self.manifest.iloc[idx]
        file_path = row["file"]; win = row["win"]
        key       = f"{file_path}_{win}"

        # ----- caching --------------------------------------------------------
        if key in self.signal_cache:
            sigs = self.signal_cache[key].copy(); self.cache_hits += 1
        else:
            sigs = self._load_signals(file_path, win)
            if len(self.signal_cache) < self.cache_size:
                self.signal_cache[key] = sigs.copy()
            self.cache_misses += 1

        # ----- split modalities ----------------------------------------------
        ecg = sigs[self.ecg_channel:self.ecg_channel+1]     # (1,L) or (>1,L)
        psg = sigs[self.psg_channels]                       # (7,L)

        # ----- augmentation ---------------------------------------------------
        if self.augment:
            ecg, psg = self._apply_augmentation(ecg, psg)

        # ----- derive extra leads --------------------------------------------
        derived = self._compute_derived_leads(ecg).astype(np.float32)  # (4,L)
        clinical = self._get_clinical_features(idx)

        return {
            "ecg_target"       : torch.from_numpy(ecg).float(),       # (1,L)
            "psg_conditioning" : torch.from_numpy(psg).float(),       # (7,L)
            "derived_leads"    : torch.from_numpy(derived).float(),   # (4,L)
            "clinical_features": torch.from_numpy(clinical).float(),  # (F,)
            "metadata"         : {
                "file"   : file_path,
                "window" : win,
                "patient": row.get("ParticipantKey", "unknown"),
                "split"  : self.split,
            }
        }

    # ---------- helper: compute derived leads --------------------------------
    def _compute_derived_leads(self, ecg: np.ndarray) -> np.ndarray:
        """
        Leads III, aVR, aVL, aVF derived from leads I & II.
        Falls back gracefully if only one lead present.
        """
        if ecg.shape[0] >= 2:
            lead_I, lead_II = ecg[0], ecg[1]
        else:                               # only lead II
            lead_II         = ecg[0]
            lead_I          = np.zeros_like(lead_II)

        lead_III = lead_II - lead_I
        aVR      = -(lead_I + lead_II) / 2.0
        aVL      =  lead_I - lead_II / 2.0
        aVF      =  lead_II - lead_I / 2.0
        return np.stack([lead_III, aVR, aVL, aVF], axis=0)  # (4,L)

    # ---------- signal loading ------------------------------------------------
    def _load_signals(self, file_path: str, window_idx: int) -> np.ndarray:
        try:
            stem = Path(file_path).stem
            npz_try = [
                Path(OUTPUT_DIR)/"edf_windows"/f"{stem}__w{window_idx}.npz",
                Path(OUTPUT_DIR)/"windows"/f"{stem}.npz",
                Path(file_path).with_suffix(".npz"),
            ]
            npz_file = next((p for p in npz_try if p.exists()), None)
            if npz_file is None:
                raise FileNotFoundError

            with np.load(npz_file, allow_pickle=True) as data:
                for k in ["data", "arr", "arr_0", "signals"]:
                    if k in data:
                        sigs = data[k]; break
                else:
                    raise KeyError("no signal key")

            if sigs.ndim == 3: sigs = sigs[window_idx]
            if sigs.ndim != 2: raise ValueError

            # pad / trim channels ---------------------------------------------
            if sigs.shape[0] < self.expected_ch:
                tmp = np.zeros((self.expected_ch, sigs.shape[1]))
                tmp[:sigs.shape[0]] = sigs; sigs = tmp
            elif sigs.shape[0] > self.expected_ch:
                sigs = sigs[:self.expected_ch]

            # resample ---------------------------------------------------------
            if sigs.shape[1] != self.window_samples:
                resamp = np.zeros((sigs.shape[0], self.window_samples))
                for ch in range(sigs.shape[0]):
                    resamp[ch] = resample(sigs[ch], self.window_samples)
                sigs = resamp

            # filtering --------------------------------------------------------
            if self.config["filter_ecg"]:
                b, a = self.ecg_ba
                sigs[self.ecg_channel] = filtfilt(b, a, sigs[self.ecg_channel])
            if self.config["filter_psg"]:
                b, a = self.psg_ba
                for ch in self.psg_channels:
                    sigs[ch] = filtfilt(b, a, sigs[ch])

            # normalise --------------------------------------------------------
            return self._normalize_signals(sigs).astype(np.float32)

        except Exception as e:
            print(f"⚠️ Error loading {file_path}, win {window_idx}: {e}")
            return np.zeros((self.expected_ch, self.window_samples), dtype=np.float32)

    # ---------- normalisation -------------------------------------------------
    def _normalize_signals(self, sigs: np.ndarray) -> np.ndarray:
        # ---- global z‑score if provided ----
        if self.gμ is not None:
            return (sigs - self.gμ[:, None]) / (self.gσ[:, None] + 1e-8)
        # ---- fallback: old min‑max ----------
        for ch in range(sigs.shape[0]):
            p1, p99 = np.percentile(sigs[ch], [1, 99])
            sigs[ch] = np.clip(sigs[ch], p1, p99)
            mn, mx   = sigs[ch].min(), sigs[ch].max()
            sigs[ch] = (sigs[ch] - mn) / (mx - mn + 1e-8) if mx > mn else 0.5
            sigs[ch] = np.clip(sigs[ch], 1e-6, 1-1e-6)
        return sigs


    # ---------- clinical helper ----------------------------------------------
    def _get_clinical_features(self, idx: int) -> np.ndarray:
        if self.clinical_data is None:
            return np.zeros(self.config["clinical_features"], dtype=np.float32)

        row  = self.manifest.iloc[idx]
        clin = self.clinical_data
        match = clin[(clin["file"] == row["file"]) & (clin["win"] == row["win"])]

        cols = [c for c in clin.columns if c not in ["file", "win"]]
        if match.empty:
            return clin[cols].mean().values.astype(np.float32)
        return match.iloc[0][cols].values.astype(np.float32)

    # ---------- augmentation --------------------------------------------------
    def _apply_augmentation(self, ecg: np.ndarray, psg: np.ndarray):
        ecg, psg = ecg.copy(), psg.copy(); cfg = self.aug_cfg; L = ecg.shape[1]

        # time‑shift ----------------------------------------------------------
        if np.random.rand() < cfg["time_shift_prob"]:
            max_s = int(cfg["time_shift_max"] * L)
            s     = np.random.randint(-max_s, max_s + 1)
            ecg = np.roll(ecg, s, axis=1); psg = np.roll(psg, s, axis=1)

        # amplitude scale -----------------------------------------------------
        if np.random.rand() < cfg["amp_scale_prob"]:
            ecg *= np.random.uniform(*cfg["amp_scale_range"])
            for ch in range(psg.shape[0]):
                psg[ch] *= np.random.uniform(*cfg["amp_scale_range"])
            ecg = np.clip(ecg, 0, 1); psg = np.clip(psg, 0, 1)

        # noise ---------------------------------------------------------------
        if np.random.rand() < cfg["noise_prob"]:
            lvl = cfg["noise_level"]
            ecg += np.random.randn(*ecg.shape) * lvl * np.std(ecg)
            psg += np.random.randn(*psg.shape) * lvl * np.std(psg, axis=1, keepdims=True)
            ecg = np.clip(ecg, 0, 1); psg = np.clip(psg, 0, 1)

        # channel dropout -----------------------------------------------------
        if np.random.rand() < cfg["channel_dropout_prob"]:
            max_d = min(cfg["channel_dropout_max"], psg.shape[0] - 1)
            k     = np.random.randint(1, max_d + 1)
            drop  = np.random.choice(psg.shape[0], k, replace=False)
            psg[drop] = 0

        return ecg, psg

    # ---------- cache stats ---------------------------------------------------
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        return dict(
            cache_size = len(self.signal_cache),
            max_cache  = self.cache_size,
            hits       = self.cache_hits,
            misses     = self.cache_misses,
            hit_rate   = self.cache_hits / total if total else 0
        )

    
    
    
    
STATS_FILE = OUTPUT_DIR / "global_stats.npz"

if not STATS_FILE.exists():
    tmp = SleepECGData(str(MANIFEST_CSV), split="train",
                       augment=False, cache_size=0,
                       preload_clinical=False)
    ch   = tmp.expected_ch
    mean = np.zeros(ch); m2 = np.zeros(ch)
    for i in tqdm(range(len(tmp)), desc="μ/σ pass"):
        sig = tmp._load_signals(tmp.manifest.loc[i,"file"],
                                tmp.manifest.loc[i,"win"])
        mean += sig.mean(1)
        m2   += (sig**2).mean(1)
    mean /= len(tmp)
    std   = np.sqrt(m2/len(tmp) - mean**2)
    np.savez(STATS_FILE, mean=mean, std=std)
else:
    stats = np.load(STATS_FILE)
    mean, std = stats["mean"], stats["std"]

GLOBAL_MEAN, GLOBAL_STD = mean, std
print("✓ global z‑score stats loaded")
#───────────────────────────────────────────────────────────────
    
# ================================================================
# 15.2  Balanced Sampling Strategy
# ================================================================
def create_balanced_sampler(dataset: SleepECGData, 
                          strategy: str = "patient") -> WeightedRandomSampler:
    """Create a weighted sampler for balanced training"""
    
    if strategy == "patient":
        # Balance by patient
        patient_counts = dataset.manifest['ParticipantKey'].value_counts()
        patient_weights = 1.0 / patient_counts
        sample_weights = dataset.manifest['ParticipantKey'].map(patient_weights)
        
    elif strategy == "severity":
        # Balance by AHI severity if available
        if 'log_AHI' in dataset.manifest.columns:
            # Create AHI severity bins
            ahi_bins = pd.qcut(dataset.manifest['log_AHI'], q=4, labels=['mild', 'moderate', 'severe', 'critical'])
            severity_counts = ahi_bins.value_counts()
            severity_weights = 1.0 / severity_counts
            sample_weights = ahi_bins.map(severity_weights)
        else:
            print("⚠️ No AHI data for severity balancing, using uniform weights")
            sample_weights = torch.ones(len(dataset))
    
    elif strategy == "combined":
        # Combine patient and severity balancing
        if 'log_AHI' in dataset.manifest.columns:
            # Patient balance
            patient_counts = dataset.manifest['ParticipantKey'].value_counts()
            patient_weights = 1.0 / patient_counts
            patient_sample_weights = dataset.manifest['ParticipantKey'].map(patient_weights)
            
            # Severity balance
            ahi_bins = pd.qcut(dataset.manifest['log_AHI'], q=4, labels=['mild', 'moderate', 'severe', 'critical'])
            severity_counts = ahi_bins.value_counts()
            severity_weights = 1.0 / severity_counts
            severity_sample_weights = ahi_bins.map(severity_weights)
            
            # Combine weights
            sample_weights = patient_sample_weights * severity_sample_weights
        else:
            sample_weights = dataset.manifest['ParticipantKey'].value_counts()
            sample_weights = 1.0 / sample_weights
            sample_weights = dataset.manifest['ParticipantKey'].map(sample_weights)
    
    else:
        # Uniform sampling
        sample_weights = torch.ones(len(dataset))
    
    # Normalize weights
    sample_weights = torch.tensor(sample_weights.values, dtype=torch.float)
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ================================================================
# 15.3  Data Loader Factory
# ================================================================
def create_ecg_reconstruction_dataloaders(
    manifest_csv: str,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    augment_train: bool = True,
    sampling_strategy: str = "patient",
    cache_size: int = 1000,
    persistent_workers=True
) -> Dict[str, DataLoader]:
    """Create data loaders for ECG reconstruction training"""
    
    print(f"🏭 Creating ECG reconstruction data loaders...")
    print(f"   Batch size: {batch_size}")
    print(f"   Workers: {num_workers}")
    print(f"   Sampling strategy: {sampling_strategy}")
    print(f"   Cache size: {cache_size}")
    
    # Create datasets
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n📂 Creating {split} dataset...")
        
        datasets[split] = SleepECGData(
            manifest_csv=manifest_csv,
            split=split,
            augment=augment_train and split == 'train',
            cache_size=cache_size if split == 'train' else cache_size // 4,
            global_mean=GLOBAL_MEAN,          #  ← NEW
            global_std =GLOBAL_STD            #  ← NEW
        )

        
        # Create sampler for training
        if split == 'train' and sampling_strategy != "uniform":
            sampler = create_balanced_sampler(datasets[split], sampling_strategy)
            shuffle = False
        else:
            sampler = None
            shuffle = split == 'train'
        
        # Create data loader
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=split == 'train',
            persistent_workers=num_workers > 0
        )
        
        print(f"   ✅ {split}: {len(datasets[split]):,} samples, {len(dataloaders[split])} batches")
    
    return dataloaders, datasets

# ================================================================
# 15.4  Data Pipeline Validation
# ================================================================
def validate_data_pipeline(dataloaders: Dict[str, DataLoader], 
                          datasets: Dict[str, SleepECGData]):
    """Validate the data loading pipeline"""
    
    print(f"\n🔍 VALIDATING DATA PIPELINE")
    print(f"=" * 50)
    
    validation_results = {}
    
    for split, dataloader in dataloaders.items():
        print(f"\n📊 Validating {split} data loader...")
        
        # Test a few batches
        batch_times = []
        signal_stats = []
        
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Test first 5 batches
                break
            
            start_time = time.time()
            
            ecg_target = batch['ecg_target']
            psg_conditioning = batch['psg_conditioning'] 
            clinical_features = batch['clinical_features']
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            derived = batch['derived_leads']
            # Collect statistics
            signal_stats.append({
                'ecg_mean': ecg_target.mean().item(),
                'ecg_std': ecg_target.std().item(),
                'ecg_min': ecg_target.min().item(),
                'ecg_max': ecg_target.max().item(),
                'psg_mean': psg_conditioning.mean().item(),
                'psg_std': psg_conditioning.std().item(),
                'clinical_mean': clinical_features.mean().item(),
                'clinical_std': clinical_features.std().item(),
                'derived_mean': derived.mean().item(),
            'derived_std' : derived.std().item(),
            })
            
            # Validate shapes
            batch_size = ecg_target.shape[0]
            
            print(f"   Batch {i+1}:")
            print(f"     ECG target: {tuple(ecg_target.shape)}")
            print(f"     PSG conditioning: {tuple(psg_conditioning.shape)}")
            print(f"     Clinical features: {tuple(clinical_features.shape)}")
            print(f"     Load time: {batch_time:.3f}s")
        
        # Calculate averages
        avg_batch_time = np.mean(batch_times)
        avg_stats = {key: np.mean([s[key] for s in signal_stats]) for key in signal_stats[0]}
        
        validation_results[split] = {
            'avg_batch_time': avg_batch_time,
            'signal_stats': avg_stats,
            'cache_stats': datasets[split].get_cache_stats()
            
        }
        
        print(f"\n   📈 {split.upper()} Statistics:")
        print(f"     Average batch time: {avg_batch_time:.3f}s")
        print(f"     ECG signal range: [{avg_stats['ecg_min']:.3f}, {avg_stats['ecg_max']:.3f}]")
        print(f"     PSG signal std: {avg_stats['psg_std']:.3f}")
        print(f"     Derived leads      : {tuple(derived.shape)}")

        
        if split == 'train':
            cache_stats = datasets[split].get_cache_stats()
            
            print(f"     Cache hit rate: {cache_stats['hit_rate']:.1%}")
    
    return validation_results




#================================================================
# 15.5  Visualization Functions
# ================================================================
def visualize_batch_samples(dataloader: DataLoader, num_samples: int = 3):
    """Visualize samples from a batch"""
    
    print(f"\n🎨 VISUALIZING BATCH SAMPLES")
    print(f"=" * 40)
    
    # Get a batch
    batch = next(iter(dataloader))
    ecg_target = batch['ecg_target']
    psg_conditioning = batch['psg_conditioning']
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, ecg_target.shape[0])):
        # Plot ECG target
        axes[i, 0].plot(ecg_target[i, 0].numpy())
        axes[i, 0].set_title(f'Sample {i+1}: ECG Target')
        axes[i, 0].set_xlabel('Time (samples)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True)
        
        # Plot PSG conditioning signals
        psg_data = psg_conditioning[i].numpy()
        for ch in range(min(3, psg_data.shape[0])):  # Plot first 3 PSG channels
            axes[i, 1].plot(psg_data[ch], label=f'PSG Ch {ch+1}', alpha=0.7)
        
        axes[i, 1].set_title(f'Sample {i+1}: PSG Conditioning')
        axes[i, 1].set_xlabel('Time (samples)')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = OUTPUT_DIR / "ecg_reconstruction_samples.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"📊 Sample visualization saved: {viz_path}")
    
    return fig

def plot_augmentation_examples(dataset: SleepECGData):
    """Show examples of data augmentation"""
    
    if not dataset.augment:
        print("⚠️ Augmentation not enabled for this dataset")
        return None
    
    print(f"\n🔄 AUGMENTATION EXAMPLES")
    print(f"=" * 30)
    
    # Get original sample
    dataset.augment = False
    original = dataset[0]
    
    # Get augmented samples
    dataset.augment = True
    augmented_samples = [dataset[0] for _ in range(4)]
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 12))
    
    # Plot original
    axes[0, 0].plot(original['ecg_target'][0].numpy())
    axes[0, 0].set_title('Original ECG')
    axes[0, 1].plot(original['psg_conditioning'][0].numpy(), alpha=0.7)
    axes[0, 1].set_title('Original PSG (Ch 1)')
    
    # Plot augmented versions
    for i, aug_sample in enumerate(augmented_samples):
        axes[i+1, 0].plot(aug_sample['ecg_target'][0].numpy())
        axes[i+1, 0].set_title(f'Augmented ECG {i+1}')
        axes[i+1, 1].plot(aug_sample['psg_conditioning'][0].numpy(), alpha=0.7)
        axes[i+1, 1].set_title(f'Augmented PSG {i+1}')
    
    for ax in axes.flat:
        ax.grid(True)
        ax.set_xlabel('Time (samples)')
    
    plt.tight_layout()
    
    # Save visualization
    aug_viz_path = OUTPUT_DIR / "augmentation_examples.png"
    plt.savefig(aug_viz_path, dpi=150, bbox_inches='tight')
    print(f"📊 Augmentation examples saved: {aug_viz_path}")
    
    return fig

# ================================================================
# 15.6  Initialize Data Pipeline
# ================================================================

# Configuration
config = {
    'batch_size': 32,
    'num_workers': 8,
    'pin_memory': True,
    'augment_train': True,
    'sampling_strategy': 'patient',  # 'patient', 'severity', 'combined', 'uniform'
    'cache_size': 500# Reduced for memory efficiency
}

# Paths
MANIFEST_CSV = OUTPUT_DIR / "manifest_cnvae_ready.csv"

print(f"📋 Configuration:")
for key, value in config.items():
    print(f"   {key}: {value}")

# Create data loaders
try:
    if MANIFEST_CSV.exists():
        dataloaders, datasets = create_ecg_reconstruction_dataloaders(
            manifest_csv=str(MANIFEST_CSV),
            **config
        )
        
        # Validate pipeline
        validation_results = validate_data_pipeline(dataloaders, datasets)
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        
        # Visualize batch samples
        if 'train' in dataloaders:
            sample_fig = visualize_batch_samples(dataloaders['train'], num_samples=3)
            plt.show()
        
        # Show augmentation examples
        if 'train' in datasets and datasets['train'].augment:
            aug_fig = plot_augmentation_examples(datasets['train'])
            if aug_fig:
                plt.show()
        
        # Memory usage check
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"\n💾 Memory usage: {memory_mb:.1f} MB")
        
        # Save data loader configuration
        loader_config = {
            'config': config,
            'validation_results': validation_results,
            'datasets_info': {
                split: {
                    'length': len(dataset),
                    'patients': dataset.manifest['ParticipantKey'].nunique(),
                    'augmentation': dataset.augment
                }
                for split, dataset in datasets.items()
            }
        }
        
        config_path = OUTPUT_DIR / "dataloader_config.json"
        with open(config_path, 'w') as f:
            json.dump(loader_config, f, indent=2, default=str)
        
        print(f"\n💾 Data loader configuration saved: {config_path}")
        print(f"✅ Data pipeline ready for cNVAE-ECG training!")
        
        # Clean up visualization memory
        plt.close('all')
        gc.collect()
        
    else:
        print(f"❌ Manifest file not found: {MANIFEST_CSV}")
        print(f"   Please run Cell 12 first to create the enhanced manifest")
        
except Exception as e:
    print(f"❌ Error creating data loaders: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🏁 Cell 15 complete!")
print(f"📊 Multi-modal data pipeline ready for ECG reconstruction training")


# ===============================================================
# 16 ▸ cNVAE‑ECG TRAINING PIPELINE
#     PSG‑to‑ECG reconstruction w/ sleep‑prior network
# ===============================================================


import os, sys, gc, json, warnings
from pathlib import Path
from datetime import datetime

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter    #  ← NEW

from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm.auto import tqdm
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────────────────────────────────────────
# 16.0  Hard reload & memory scrub
# ───────────────────────────────────────────────────────────────
for m in [m for m in sys.modules if 'conditional' in m or 'neural_operations' in m]:
    del sys.modules[m]
for p in [p for p in sys.path if 'conditional' in p or 'cNVAE_ECG' in p]:
    sys.path.remove(p)
print("🔄  Module cache cleared")
torch.cuda.empty_cache(); gc.collect()

# ───────────────────────────────────────────────────────────────
# 16.0.1  Import library patches
# ───────────────────────────────────────────────────────────────
parent_path = "/home/jupyter/cnvae-ecg-us-central1-xdgfm/cNVAE_ECG-main"
sys.path[:0] = [parent_path, f"{parent_path}/conditional"]

import neural_operations_1d
neural_operations_1d.SYNC_BN = False
neural_operations_1d.get_batchnorm = lambda *a, **k: nn.BatchNorm1d(*a, **k)

from conditional.model_conditional_1d import AutoEncoder
from conditional import utils

FAST_ADAMAX = False
from torch.optim import Adam as Adamax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")
# ───────────────────────────────────────────────────────────────
# 16.0.2  FixedAutoEncoder (adds the sleep‑prior MLP)
# ───────────────────────────────────────────────────────────────
class _SqueezeEmbed(nn.Module):
    """(B,1,H,L) ➜ squeeze ➜ Conv1d ➜ GAP ➜ (B, C_enc, 1)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 1, H, L)  →  (B, H, L)
        x = x.squeeze(1)
        x = self.conv(x)
        return self.pool(x)

class FixedAutoEncoder(AutoEncoder):
    """
    cNVAE‑ECG backbone with an extra MLP that embeds the clinical
    sleep vector and feeds it in as the conditional input.
    """
    def __init__(self, args, writer, arch_instance, num_classes):
        super().__init__(args, writer, arch_instance, num_classes)

        # --- replace embed_transform so it can handle 4‑D input ---
        c_scaling         = 2 ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        expected_channels = int(c_scaling * self.num_channels_dec)      # → 64
        self.embed_transform = _SqueezeEmbed(expected_channels,
                                             self.num_channels_enc)

        # Tiny 2‑layer prior network for sleep features
        self.sleep_prior = nn.Sequential(
            nn.Linear(args.clin_feat_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128),               nn.ReLU(inplace=True)
        )

    def forward(self, x, sleep_vec=None):
        if sleep_vec is None:                      # keep parent behaviour
            return super().forward(x)

        # Embed clinical vector and pass positionally
        sleep_emb = self.sleep_prior(sleep_vec)    # (B,128)
        return super().forward(x, sleep_emb)
    # ─────────────────────────────────────────────
    # ADD / RESTORE THIS METHOD
    # (B, C*2, L)  →  Normal(μ, σ)
    def decoder_output(self, logits):
        mu, log_sigma = torch.chunk(logits, 2, dim=1)
        return torch.distributions.Normal(mu, log_sigma.exp())
    
    
# ---------- correlation helper (if utils lacks it) ----------
if not hasattr(utils, "batch_corrcoef"):
    def _batch_corrcoef(x, y, eps=1e-8):
        """
        x,y shape: (B, L) → returns (B,) Pearson r
        """
        vx = x - x.mean(dim=1, keepdim=True)
        vy = y - y.mean(dim=1, keepdim=True)
        num = (vx * vy).mean(dim=1)
        den = (vx.var(dim=1, unbiased=False).sqrt() *
               vy.var(dim=1, unbiased=False).sqrt() + eps)
        return num / den
    utils.batch_corrcoef = _batch_corrcoef

# ───────────────────────────────────────────────────────────────
# 16.1  Training hyper‑parameters  (FULL replacement)
# ───────────────────────────────────────────────────────────────
class TrainingArgs:
    def __init__(self):
        # — Architecture —
        self.arch_instance          = 'res_bnelu'
        self.num_channels_enc       = 8
        self.num_channels_dec       = 32
        self.num_latent_scales      = 2
        self.num_groups_per_scale   = 10
        self.num_latent_per_group   = 4
        self.num_preprocess_blocks  = 2
        self.num_preprocess_cells   = 2      #  << NEW
        self.num_postprocess_blocks = 2
        self.num_postprocess_cells  = 2      #  << NEW
        self.num_cell_per_cond_enc  = 1      #  << NEW
        self.num_cell_per_cond_dec  = 1      #  << NEW
        self.num_mixture_dec        = 1   #  discrete-log? normal?
        self.num_nf                 = 0      #  << NEW  (normalising flows disabled)
        self.num_input_channels     = 7     #   7 PSG 
        self.input_size             = 1920   #  15 s @ 128 Hz

        # — Misc flags NVAE expects —
        self.use_se                 = False  #  << NEW
        self.res_dist               = False  #  << NEW
        self.ada_groups             = False  #  << NEW
        self.min_groups_per_scale   = 1      #  << NEW
        self.num_x_bits             = 8      #  << NEW (bit‑depth for DiscMix)

        # — Clinical vector —
        self.clin_feat_dim          = 5

        # — Extra focal‑loss switches —
        self.focal                  = False
        self.focal_alpha            = 0.0
        self.focal_gamma            = 0.0

        # — Training —
        self.batch_size             = 32
        self.epochs                 = 200
        self.learning_rate          = 1e-3
        self.learning_rate_min      = 5e-4
        self.weight_decay           = 0
        self.weight_decay_norm      = 1e-2
        self.weight_decay_norm_init = 10.0
        self.kl_anneal_portion      = 0.3
        self.kl_const_portion       = 0.0001
        self.kl_const_coeff         = 0.0001
        self.percent_epochs         = 5
        self.fast_adamax            = FAST_ADAMAX
        self.seed                   = 1
        self.use_amp                = True
        self.grad_accum             = 1

        # — Book‑keeping —
        run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path("outputs/cnvae_checkpoints") / f"run_{run_ts}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tb_root   = Path("outputs/tb_runs")
        self.tb_root.mkdir(parents=True, exist_ok=True)
        self.tb_logdir = self.tb_root / f"run_{run_ts}"      # NEW


    # Safety‑net: fall back to None for any other optional flags
    def __getattr__(self, item):          # called only if attribute missing
        return None

args = TrainingArgs()
torch.manual_seed(args.seed); np.random.seed(args.seed)

# ───────────────────────────────────────────────────────────────
# 16.2  Data (expects channel order: ECG | 7 PSG | 4 derived)
# ───────────────────────────────────────────────────────────────
if 'dataloaders' not in globals():
    raise RuntimeError("🔴  Please run Cell 15 first so that `dataloaders` are defined.")

train_dataset = dataloaders['train'].dataset
valid_dataset = dataloaders['val'].dataset

# Auto‑detect real clinical‑feature dimension
args.clin_feat_dim = int(train_dataset[0]['clinical_features'].shape[0])
print(f"ℹ️ Clinical‑feature dim detected → {args.clin_feat_dim}")

#from torch.utils.data import DataLoader
#train_queue = DataLoader(train_dataset, batch_size=args.batch_size,
#                         shuffle=True,  num_workers=8, pin_memory=True,persistent_workers = True, drop_last=True)
#valid_queue = DataLoader(valid_dataset, batch_size=args.batch_size,
#                         shuffle=False, num_workers=8, #pin_memory=True,persistent_workers = True, drop_last=True)

#train_queue = dataloaders["train"]   # reuse existing loader
#valid_queue = dataloaders["val"]






from torch.utils.data import DataLoader, Subset

BATCH_SIZE = 32

train_ds_full = SleepECGData(str(MANIFEST_CSV),
                             split="train",
                             augment=False,          # 🔑 OFF
                             cache_size=0,
                             global_mean=GLOBAL_MEAN,
                             global_std=GLOBAL_STD)

tiny_idx   = list(range(BATCH_SIZE))      # first 32 windows
tiny_train = Subset(train_ds_full, tiny_idx)

train_queue = DataLoader(tiny_train,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0)
valid_queue = train_queue                 # same batch
print("✓ tiny loaders ready →", len(train_queue), "batches")






args.num_total_iter = len(train_queue) * args.epochs

# ───────────────────────────────────────────────────────────────
# 16.3  Model, optimiser, scheduler
# ───────────────────────────────────────────────────────────────
arch_cells = utils.get_arch_cells(args.arch_instance)
model = FixedAutoEncoder(args=args, writer=None,
                         arch_instance=arch_cells,
                         num_classes=128).to(device)

opt_cls   = Adamax
optimizer = opt_cls(model.parameters(), lr=args.learning_rate,
                    weight_decay=args.weight_decay, eps=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=float(args.epochs - np.floor(args.epochs * args.percent_epochs / 100) - 1),
    eta_min=args.learning_rate_min
)
scaler = GradScaler(enabled=args.use_amp) 
# disable for the sanity test
autocast_ctx = autocast(enabled=args.use_amp) if device.type == "cuda" else nullcontext()


tb_writer = SummaryWriter(log_dir=str(args.tb_logdir))    #  ← NEW
print(f"📊 TensorBoard logs → {args.tb_logdir}")


# ───────────────────────────────────────────────────────────────
# 16.4  Training & validation helpers
# ───────────────────────────────────────────────────────────────
@torch.no_grad()
def _model_input(batch):
    """Return ONLY the 7‑ch PSG block (B,7,L)"""
    return batch['psg_conditioning']



def train_epoch(loader, global_step, tb_writer):
    model.train()
    alpha = utils.kl_balancer_coeff(model.num_latent_scales,
                                    model.groups_per_scale, fun='square')
    meter = utils.AvgrageMeter()
    accum = args.grad_accum
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        psg    = _model_input(batch).to(device, non_blocking=True)
        target = batch['ecg_target'].to(device, non_blocking=True)
        sleep  = batch['clinical_features'].to(device, non_blocking=True)

        with autocast_ctx:
            logits, *_= model(psg, sleep_vec=sleep)
            # negative log‑likelihood from the logistic mixture
            dist = model.decoder_output(logits) # Normal(B,C,L)
            recon_loss = -dist.log_prob(target).sum([1, 2]) # NLL for each sample
            
            mu = model.decoder_output(logits).mean      # keep handy for corr penalty
            
            corr = 1.0 - utils.batch_corrcoef(
                        mu.view(mu.size(0), -1),       # flatten
                        target.view(target.size(0), -1)
                    ).mean()
            aux_w = 0.2    
            #kl_coeff   = utils.kl_coeff(global_step,
            #                            args.kl_anneal_portion * args.num_total_iter,
            #                            args.kl_const_portion  * args.num_total_iter,
            #                            args.kl_const_coeff)
            #balanced_kl, _, _ = utils.kl_balancer(rest[2], kl_coeff, False, alpha)
            
            #loss = (recon_loss + balanced_kl).mean() / accum
            kl_coeff     = 0.0                 # β = 0
            balanced_kl  = 0.0
            loss = recon_loss.mean()
        scaler.scale(loss).backward()
        # step every `accum` mini‑batches
        if i % accum == 0 or i == len(loader):
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        meter.update(loss.item()*accum, target.size(0))

        if global_step % 20 == 0:
            tb_writer.add_scalar("step/loss", loss.item()*accum, global_step)
            tb_writer.add_scalar("step/kl_coeff", kl_coeff, global_step)
        global_step += 1

    return meter.avg, global_step

@torch.no_grad()
def validate(loader, tb_writer, epoch):
    model.eval()
    meter = utils.AvgrageMeter()
    corr_vals = []
    alpha = utils.kl_balancer_coeff(model.num_latent_scales,
                                model.groups_per_scale, fun='square')

    for batch in tqdm(loader, desc="Val  ", leave=False):
        psg    = _model_input(batch).to(device, non_blocking=True)
        target = batch['ecg_target'].to(device, non_blocking=True)
        sleep  = batch['clinical_features'].to(device, non_blocking=True)

        logits, *rest = model(psg, sleep_vec=sleep)
        dist = model.decoder_output(logits) # Normal(B,C,L)
        recon_loss = -dist.log_prob(target).sum([1, 2]) # NLL for each sample
       # balanced_kl, _, _ = utils.kl_balancer(rest[2], 1.0, False, alpha)
       # loss = (recon_loss + balanced_kl).mean()
        balanced_kl = 0.0          # disable KL
        loss = recon_loss.mean()
        meter.update(loss.item(), target.size(0))

        # ---------- Pearson r on lead II ----------
        pred    = model.decoder_output(logits).mean[:, 0]   # (B,L)
        r_batch = utils.batch_corrcoef(pred, target[:,0])   # (B,)
        corr_vals.append(r_batch.mean().item())

    r_epoch = float(np.mean(corr_vals))
    tb_writer.add_scalar("val/NELBO", meter.avg, epoch)
    tb_writer.add_scalar("val/r_corr_leadII", r_epoch, epoch)   # NEW

    return meter.avg, r_epoch

# ───────────────────────────────────────────────────────────────
# 16.5  Main training loop
# ───────────────────────────────────────────────────────────────
best_val, global_step = np.inf, 0
history = {"train": [], "val": []}

for epoch in range(args.epochs):
    print(f"\n🌀 Epoch {epoch+1:02d}/{args.epochs}")
    if epoch >= np.floor(args.epochs * args.percent_epochs / 100):
        scheduler.step()

    train_loss, global_step = train_epoch(train_queue, global_step,tb_writer)
    val_loss,val_r          = validate(valid_queue,tb_writer,epoch)
    # --- TensorBoard: epoch‑level ---
    tb_writer.add_scalar("epoch/train_NELBO", train_loss, epoch)
    tb_writer.add_scalar("epoch/val_NELBO",   val_loss,   epoch)
    tb_writer.flush()

    history["train"].append(float(train_loss))
    history["val"].append(float(val_loss))
    print(f"   ➜ train NELBO ≈ {train_loss:.4f} | "
          f"val NELBO ≈ {val_loss:.4f}  |  r ≈ {val_r:.3f}")

    # Save best checkpoint
    if val_loss < best_val:
        best_val = val_loss
        ckpt_path = args.save_dir / "model_best.pt"
        torch.save({"state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_loss": best_val}, ckpt_path)
        print(f"   ✅ New best model saved → {ckpt_path}")

# ───────────────────────────────────────────────────────────────
# 16.6  Persist training curve
# ───────────────────────────────────────────────────────────────
hist_file = args.save_dir / "training_history.json"
with open(hist_file, "w") as f:
    json.dump(history, f, indent=2)
tb_writer.close()               #  ← NEW
print(f"\n📈 Loss history saved → {hist_file}")
print("\n🏁 Training complete!")



    ############################################################

