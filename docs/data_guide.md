# Data Usage Guide

This guide explains how to work with the Sunnybrook Sleep Laboratory dataset for the cNVAE-PSG project.

## Dataset Overview

The project uses the **Comprehensive Sleep Laboratory Data: August - October 2024** from Sunnybrook Health Sciences Centre, available through the Health Data Nexus platform.

### Dataset Citation
```
Berger, S., Boulos, M., Tchoudnovski, D., Byeon, A., Tandon, A., & Murray, B. (2024). 
Comprehensive Sleep Laboratory Data: August - October 2024 (version 1.0). 
Health Data Nexus. https://doi.org/10.57764/tvsv-y363.
```

## Data Access Requirements

### Prerequisites
1. **Institutional Affiliation**: Must be affiliated with a research institution
2. **Ethics Training**: Complete Health Data Nexus Data User Code of Conduct Training
3. **Data Use Agreement**: Sign the T-CAIREM Data Use Agreement
4. **REB Approval**: Ensure your research has appropriate ethics board approval

### Access Process
1. **Register** on the Health Data Nexus platform
2. **Complete training** requirements
3. **Submit access request** for the dataset
4. **Sign agreements** and provide institutional verification
5. **Download data** through the research environment

## Data Structure

### EDF Files (Polysomnography Data)
Located in: `gcs/EDF_Files/`

**Signal Channels:**
- **EEG**: C3-A2, C4-A1, O1-A2, O2-A1
- **EOG**: LOC-A2, ROC-A1  
- **EMG**: EMG1-EMG2 (chin)
- **ECG**: ECG1, ECG2
- **Respiratory**: Airflow (nasal), Chest effort, Abdominal effort
- **Other**: Body position, SpO2, leg movements

**File Specifications:**
- Format: European Data Format (EDF)
- Sampling rates: Variable by channel (typically 250-500 Hz)
- Duration: Full overnight recordings (~8 hours)
- File naming: Patient-specific identifiers

### Clinical Data (CSV)
Located in: `gcs/TCAIREM_SleepLabData.csv`

**Categories:**
- **Demographics**: Age, sex, BMI, height, weight
- **Sleep metrics**: Total sleep time, sleep efficiency, sleep stages
- **Clinical scores**: AHI, arousal index, sleep latency
- **Questionnaires**: Multiple validated instruments (see below)

### Data Dictionary
Located in: `Sleep Data Organization - All Data Variables.csv`

Complete variable definitions and descriptions for all clinical data fields.

## Loading Data

### Python Setup
```python
import os
from pathlib import Path
import pyedflib
import pandas as pd

# Set data root (override with environment variable)
DATA_ROOT = Path(os.environ.get("TC_DATA", "."))
EDF_DIR = DATA_ROOT / "gcs" / "EDF_Files"
CLINICAL_CSV = DATA_ROOT / "gcs" / "TCAIREM_SleepLabData.csv"
```

### Loading EDF Files
```python
from pyedflib import highlevel
import pyedflib as plib

def load_edf_file(edf_path):
    """Load EDF file and return signals, headers, and metadata."""
    signals, signal_headers, header = highlevel.read_edf(edf_path)
    return signals, signal_headers, header

# Example usage
edf_path = EDF_DIR / "patient_001.edf"
signals, signal_headers, header = load_edf_file(edf_path)

# Print available channels
for i, header in enumerate(signal_headers):
    print(f"Channel {i}: {header['label']} - {header['sample_rate']} Hz")
```

### Loading Clinical Data
```python
# Load clinical data
clinical_data = pd.read_csv(CLINICAL_CSV)

# Load data dictionary
data_dict = pd.read_csv(DATA_ROOT / "Sleep Data Organization - All Data Variables.csv")

print(f"Clinical data shape: {clinical_data.shape}")
print(f"Available variables: {clinical_data.columns.tolist()}")
```

## Data Preprocessing

### Signal Processing Pipeline
```python
import numpy as np
from scipy import signal

def preprocess_edf_signals(signals, signal_headers, target_fs=250):
    """Preprocess EDF signals for model training."""
    processed_signals = {}
    
    for i, sig in enumerate(signals):
        label = signal_headers[i]['label']
        fs = signal_headers[i]['sample_rate']
        
        # Resample to target frequency
        if fs != target_fs:
            num_samples = int(len(sig) * target_fs / fs)
            sig_resampled = signal.resample(sig, num_samples)
        else:
            sig_resampled = sig
        
        # Apply filtering
        if 'EEG' in label or 'EOG' in label:
            # EEG/EOG: 0.5-35 Hz bandpass
            sos = signal.butter(4, [0.5, 35], btype='band', fs=target_fs, output='sos')
            sig_filtered = signal.sosfilt(sos, sig_resampled)
        elif 'ECG' in label:
            # ECG: 0.5-100 Hz bandpass  
            sos = signal.butter(4, [0.5, 100], btype='band', fs=target_fs, output='sos')
            sig_filtered = signal.sosfilt(sos, sig_resampled)
        else:
            sig_filtered = sig_resampled
        
        processed_signals[label] = sig_filtered
    
    return processed_signals
```

### Patient Matching
```python
def match_edf_clinical_data(edf_dir, clinical_df):
    """Match EDF files with clinical data records."""
    matched_data = []
    
    for edf_file in edf_dir.glob("*.edf"):
        # Extract patient ID from filename
        patient_id = extract_patient_id(edf_file.name)
        
        # Find matching clinical record
        clinical_record = clinical_df[clinical_df['ID'] == patient_id]
        
        if not clinical_record.empty:
            matched_data.append({
                'patient_id': patient_id,
                'edf_path': edf_file,
                'clinical_data': clinical_record.iloc[0].to_dict()
            })
    
    return matched_data
```

## Questionnaire Data

### Available Questionnaires
1. **Medical Comorbidities**: Self-administered questionnaire
2. **STOP-BANG**: Obstructive sleep apnea risk screening
3. **Insomnia Severity Index**: Sleep disorder screening
4. **IRLS**: Restless legs syndrome severity rating
5. **MUPS**: Munich Parasomnia Screening
6. **Epworth Sleepiness Scale**: Daytime sleepiness assessment
7. **Pittsburgh Sleep Quality Index**: Sleep quality evaluation
8. **Horne-Ostberg**: Chronotype (morning/evening preference)
9. **EQ-5D-5L**: Quality of life assessment

### Processing Questionnaire Data
```python
def process_questionnaire_scores(clinical_data):
    """Extract and process questionnaire scores."""
    questionnaire_columns = [
        'epworth_score', 'psqi_score', 'stop_bang_score',
        'isi_score', 'irls_score', 'mups_score', 'ho_score'
    ]
    
    scores = clinical_data[questionnaire_columns].copy()
    
    # Handle missing values
    scores = scores.fillna(scores.median())
    
    return scores
```

## Data Privacy and Ethics

### De-identification
- All data has been de-identified according to Health Data Nexus standards
- Patient identifiers are randomized research IDs
- Direct patient identifiers have been removed

### Data Security
- Store data in secure, encrypted locations
- Never commit data files to version control
- Use secure transfer methods for data sharing
- Follow institutional data security policies

### Usage Restrictions
- **Research only**: Data cannot be used for commercial purposes
- **No re-identification**: Do not attempt to identify patients
- **Sharing restrictions**: Cannot share raw data with unauthorized users
- **Publication requirements**: Must acknowledge dataset and follow citation guidelines

## Troubleshooting

### Common Issues

**EDF Loading Errors:**
```python
# Handle corrupted or incomplete EDF files
try:
    signals, headers, metadata = load_edf_file(edf_path)
except Exception as e:
    print(f"Error loading {edf_path}: {e}")
    continue
```

**Memory Management:**
```python
# Process large files in chunks
def load_edf_chunked(edf_path, chunk_duration=300):  # 5 minutes
    """Load EDF file in chunks to manage memory."""
    # Implementation for chunked loading
    pass
```

**Missing Clinical Data:**
```python
# Handle missing clinical records
clinical_record = clinical_df[clinical_df['ID'] == patient_id]
if clinical_record.empty:
    print(f"No clinical data found for patient {patient_id}")
    # Use default values or skip
```

## Performance Optimization

### Efficient Data Loading
- Use memory mapping for large files
- Implement caching for frequently accessed data
- Process data in parallel where possible
- Use appropriate data types (float32 vs float64)

### Storage Considerations
- Consider HDF5 format for processed data
- Implement compression for storage efficiency
- Use solid-state drives for better I/O performance

## Support and Resources

### Documentation
- **PyEDFlib**: https://pyedflib.readthedocs.io/
- **Health Data Nexus**: Platform-specific documentation
- **T-CAIREM Guidelines**: Institutional data usage policies

### Contact
- **Technical Issues**: GitHub Issues
- **Data Access**: Health Data Nexus support
- **Research Questions**: Principal investigators

### Citation Requirements
When using this data in publications, cite:
1. The original dataset (Berger et al., 2024)
2. T-CAIREM funding acknowledgment
3. Sunnybrook Sleep Laboratory collaboration
4. Ethics board approval (REB #6197)
