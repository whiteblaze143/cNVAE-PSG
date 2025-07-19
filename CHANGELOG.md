# Changelog

All notable changes to the cNVAE-PSG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and GitHub preparation
- Comprehensive README with project overview and setup instructions
- Requirements.txt with all necessary dependencies
- Academic research license for T-CAIREM project
- Contributing guidelines for research collaboration
- Python package setup (setup.py) for installation
- Proper .gitignore for research projects (excluding sensitive data)

### Research Progress
- Phase 1: Basic feasibility exploration in progress
- Conditional Neural VQVAE architecture development
- PSG-to-ECG cross-modal reconstruction framework

## [0.1.0] - 2024-07-18

### Added
- Initial research codebase from T-CAIREM project
- Main training script (`cnvae_train.py`) for model development
- Jupyter notebooks for exploratory data analysis
  - `TCAIREM.ipynb`: Main research notebook with full pipeline
  - `sleep_eda.ipynb`: Sleep data exploratory analysis
- Integration with Sunnybrook Sleep Laboratory dataset
- EDF signal processing and preprocessing utilities
- Clinical data matching and patient synchronization
- Multi-channel PSG signal handling (EEG, EOG, EMG, ECG)

### Data Integration
- Support for European Data Format (EDF) files
- Clinical CSV data processing and questionnaire integration
- Memory-efficient data loading with chunked processing
- Robust patient matching between EDF and clinical data

### Model Architecture
- Conditional Neural Vector Quantized VAE (cNVAE) implementation
- Multi-modal signal preprocessing pipeline
- 12-lead ECG reconstruction target framework
- Sleep stage conditioning integration

### Documentation
- Research presentation materials
- T-CAIREM midterm presentation
- Comprehensive architecture diagrams
- Dataset organization documentation

### Research Ethics
- Sunnybrook REB #6197 compliance
- Health Data Nexus integration
- De-identification protocols
- T-CAIREM Data Use Agreement adherence

---

## Research Milestones

### Phase 1 Objectives (Current)
- [ ] Establish basic PSG-to-ECG reconstruction feasibility
- [ ] Achieve positive correlation (r > 0.2) between output and real ECG
- [ ] Document technical insights and challenges
- [ ] Validate data preprocessing pipeline

### Phase 2 Objectives (Planned)
- [ ] Improve reconstruction quality by 10-15% over Phase 1
- [ ] Investigate physiological coupling mechanisms
- [ ] Integrate sleep staging information
- [ ] Explore clinical variable conditioning

### Phase 3 Objectives (Future)
- [ ] Clinical application assessment
- [ ] Real-time processing capabilities
- [ ] Cross-institutional validation
- [ ] Publication preparation

---

## Development Notes

### Technical Decisions
- **PyTorch**: Selected for deep learning framework due to research flexibility
- **EDF Format**: Native support for polysomnography data standard
- **Jupyter Notebooks**: Chosen for exploratory research and reproducibility
- **Memory Optimization**: Implemented for large-scale sleep signal processing

### Data Considerations
- Patient privacy: All data handling follows strict de-identification protocols
- Dataset size: Comprehensive sleep lab data from August-October 2024
- Signal quality: Multi-channel PSG with clinical-grade acquisition
- Annotation richness: Sleep staging, events, and comprehensive questionnaires

### Research Reproducibility
- Fixed random seeds for model training
- Documented preprocessing steps
- Version-controlled configurations
- Environment specifications included

---

## Acknowledgments

### Research Team
- **Principal Investigator**: Dr. Christopher Cheung (Sunnybrook)
- **Data Contributors**: Sarah Berger, Mark Boulos, Dennis Tchoudnovski, Alana Byeon, Anu Tandon, Brian Murray
- **Co-Investigators**: Dr. Marc Narayansingh, Dr. Karthi Umapathy, Dr. Andrew Lim, Dr. Houman Khosravani, Vikash Nanthakumar

### Funding and Support
- **T-CAIREM**: Primary funding and institutional support
- **Sunnybrook Sleep Laboratory**: Data collection and clinical expertise
- **Health Data Nexus**: Data platform and access infrastructure

### Ethics and Compliance
- **Sunnybrook Research Ethics Board**: REB #6197 approval
- **University of Toronto**: Research oversight and guidance
- **Health Data Nexus**: Data use compliance and monitoring
