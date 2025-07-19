# cNVAE-PSG: Cross-Modal PSG-to-ECG Reconstruction

[![License: Research](https://img.shields.io/badge/License-Research-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A research project investigating cross-modal reconstruction from polysomnography (PSG) signals to electrocardiography (ECG) using conditional Neural Vector Quantized Variational Autoencoders (cNVAE).

## 🏥 Project Overview

This project explores the feasibility of reconstructing ECG signals from polysomnography data using deep learning techniques. The research is conducted as part of the **T-CAIREM** (Temerty Centre for AI Research and Education in Medicine) initiative in collaboration with the **Sunnybrook Health Sciences Centre Sleep Laboratory**.

### Research Objectives

- **Phase 1**: Establish basic feasibility of PSG-to-ECG reconstruction
- **Phase 2**: Investigate physiological coupling mechanisms between sleep signals and cardiac activity
- **Phase 3**: Explore potential clinical applications for sleep-cardiac monitoring

> **Note**: This is an exploratory study to assess cross-modal PSG-to-ECG reconstruction merit. We are establishing basic feasibility rather than promising revolutionary clinical impact.

## 📊 Dataset

This project uses the **Comprehensive Sleep Laboratory Data: August - October 2024** from the Sunnybrook Health Sciences Centre Sleep Laboratory, available through the [Health Data Nexus](https://doi.org/10.57764/tvsv-y363).

### Data Citation
```
Berger, S., Boulos, M., Tchoudnovski, D., Byeon, A., Tandon, A., & Murray, B. (2024). 
Comprehensive Sleep Laboratory Data: August - October 2024 (version 1.0). 
Health Data Nexus. https://doi.org/10.57764/tvsv-y363.
```

### Data Components
- **EDF Files**: Multi-channel polysomnography signals (EEG, EOG, EMG, ECG, respiratory, etc.)
- **Clinical Data**: Sleep metrics, questionnaires, comorbidities, medications
- **Questionnaire Battery**: STOP-BANG, Epworth Sleepiness Scale, Pittsburgh Sleep Quality Index, etc.

## 🔧 Technical Architecture

### Model: Conditional Neural Vector Quantized VAE (cNVAE)
- **Input**: Multi-channel PSG signals (EEG, EOG, EMG, respiratory)
- **Conditioning**: Sleep stage information, clinical variables
- **Output**: Reconstructed ECG signals
- **Target**: 12-lead ECG reconstruction from PSG data

### Key Features
- Memory-efficient data loading with chunked processing
- Robust patient matching between EDF and clinical data
- Multi-modal signal preprocessing and synchronization
- Advanced data augmentation for sleep signal analysis

## 📁 Project Structure

```
cNVAE-PSG/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package installation
├── .gitignore                         # Git ignore rules
├── LICENSE                            # Research license
│
├── notebooks/                         # Jupyter notebooks
│   ├── TCAIREM.ipynb                 # Main research notebook
│   └── sleep_eda.ipynb               # Exploratory data analysis
│
├── src/                              # Source code
│   ├── cnvae_train.py                # Training script
│   ├── data/                         # Data processing modules
│   ├── models/                       # Model architectures
│   └── utils/                        # Utility functions
│
├── configs/                          # Configuration files
├── experiments/                      # Experiment logs and results
├── docs/                            # Documentation
│   ├── architecture.md              # Model architecture details
│   └── data_guide.md                # Data usage guide
│
└── assets/                          # Images and presentations
    ├── cnvae_architecture_comprehensive.png
    ├── tcairem_midterm_presentation_final.pdf
    └── logos/
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Access to T-CAIREM Health Data Nexus (for dataset)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/[your-username]/cNVAE-PSG.git
   cd cNVAE-PSG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode**
   ```bash
   pip install -e .
   ```

### Data Setup

1. **Obtain dataset access** through [Health Data Nexus](https://doi.org/10.57764/tvsv-y363)
2. **Complete required training**: Health Data Nexus Data User Code of Conduct
3. **Sign Data Use Agreement**: T-CAIREM Data Use Agreement
4. **Configure data paths** in your environment:
   ```bash
   export TC_DATA="/path/to/your/tcairem/data"
   ```

### Quick Start

1. **Run Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/sleep_eda.ipynb
   ```

2. **Start Training**
   ```bash
   python src/cnvae_train.py --config configs/default.yaml
   ```

3. **Monitor Training**
   ```bash
   tensorboard --logdir experiments/logs
   ```

## 📈 Model Performance

### Success Metrics (Phase 1-2)
- **Phase 1**: Positive correlation (r > 0.2) between reconstructed and real ECG
- **Phase 2**: 10-15% improvement over Phase 1 baseline results

### Evaluation Criteria
- Signal correlation analysis
- Morphological similarity metrics
- Clinical relevance assessment
- Computational efficiency benchmarks

## 🔬 Research Context

### Clinical Hypotheses
- **H1**: Sleep stage transitions correlate with cardiac rhythm changes
- **H2**: Respiratory events during sleep influence ECG morphology
- **H3**: Physiological coupling exists between neurological and cardiac signals during sleep

### Potential Applications
- Enhanced sleep-cardiac monitoring
- Reduced sensor burden for sleep studies
- Cross-modal physiological signal analysis
- Sleep disorder cardiovascular risk assessment

## 📚 Documentation

- **[Model Architecture](docs/architecture.md)**: Detailed cNVAE implementation
- **[Data Guide](docs/data_guide.md)**: Dataset usage and preprocessing
- **[API Reference](docs/api.md)**: Code documentation
- **[Experiment Logs](experiments/)**: Training results and analysis

## 🤝 Contributing

This is an active research project. Contributions are welcome through:

1. **Issues**: Report bugs or suggest improvements
2. **Pull Requests**: Submit code improvements or new features
3. **Research Collaboration**: Contact project investigators

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the **Health Data Nexus Contributor Review Health Data License 1.0** for research and academic use only.

## 🙏 Acknowledgments

### Principal Investigator
**Dr. Christopher Cheung**  
*Principal Investigator & Research Supervisor*  
*Schulich Heart Program, Sunnybrook Health Sciences Centre*

### Funding & Support
- **T-CAIREM**: Funding & institutional support for advancing AI-driven sleep-cardiac research
- **Sleep Laboratory Team @ Sunnybrook**: Collaborative support for cross-modal dataset development


## 📞 Contact

For research inquiries and collaboration opportunities:

- **Project Lead**: [Mithun Manivannan] - [mithun.manivannan@sri.utoronto.ca]

## 📖 Citation

If you use this code or research in your work, please cite:

```bibtex
@misc{cnvae-psg-2024,
  title={Cross-Modal PSG-to-ECG Reconstruction using Conditional Neural Vector Quantized VAE},
  author={[Mithun Manivannan]},
  year={2025},
  publisher={T-CAIREM},
  url={https://github.com/[whiteblaze143]/cNVAE-PSG}
}
```

## 🔗 Related Publications

- T Kendzerska, BJ Murray, . . . MI Boulos. "Polysomnographic Assessment of Sleep Disturbances in Cancer Development: A Historical Multicenter Clinical Cohort Study." Chest. 2023.
- DR Colelli, GR Dela Cruz, . . . MI Boulos. "Impact of sleep chronotype on in-laboratory polysomnography parameters." J Sleep Res. 2023:e13922.

---

**Disclaimer**: This is experimental research software. Results are preliminary and intended for research purposes only. Not for clinical use.
