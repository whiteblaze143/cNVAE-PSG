# Contributing to cNVAE-PSG

Thank you for your interest in contributing to the cNVAE-PSG research project! This document outlines the process for contributing to this academic research codebase.

## 🎯 Project Scope

This is an active research project investigating cross-modal PSG-to-ECG reconstruction. Contributions should align with the research objectives and maintain the academic integrity of the work.

## 🤝 Types of Contributions

### Research Contributions
- Algorithm improvements and novel approaches
- Model architecture enhancements
- Evaluation metrics and analysis methods
- Performance optimizations

### Technical Contributions
- Bug fixes and error handling
- Code documentation and comments
- Unit tests and validation scripts
- Data preprocessing improvements

### Documentation Contributions
- API documentation updates
- Tutorial and example improvements
- Research methodology documentation
- Installation and setup guides

## 📋 Getting Started

### Prerequisites

1. **Academic Affiliation**: Contributors should be affiliated with academic or research institutions
2. **Data Access**: If working with data, ensure you have proper authorization through Health Data Nexus
3. **Research Ethics**: Understand and comply with research ethics requirements (REB #6197)

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/cNVAE-PSG.git
   cd cNVAE-PSG
   ```

2. **Create development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🔄 Development Workflow

### Branch Naming Convention
- `feature/description` - New features or enhancements
- `bugfix/description` - Bug fixes
- `experiment/description` - Experimental changes
- `docs/description` - Documentation updates

### Commit Messages
Follow conventional commit format:
```
type(scope): brief description

Detailed explanation if needed

- Bullet points for multiple changes
- Reference issues with #issue-number
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```bash
feat(model): add attention mechanism to cNVAE decoder
fix(data): resolve EDF loading memory leak  
docs(readme): update installation instructions
```

## 🧪 Code Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting: `black .`
- Use flake8 for linting: `flake8 src/`
- Type hints for function signatures: `mypy src/`

### Documentation Standards
- Docstrings for all public functions and classes
- Use Google-style docstrings
- Include parameter types and return values
- Provide usage examples for complex functions

**Example**:
```python
def preprocess_edf_signals(
    edf_path: Path, 
    target_fs: int = 250,
    filter_params: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """Preprocess EDF signals for model training.
    
    Args:
        edf_path: Path to the EDF file
        target_fs: Target sampling frequency in Hz
        filter_params: Optional filtering parameters
        
    Returns:
        Tuple of preprocessed signals and metadata
        
    Raises:
        FileNotFoundError: If EDF file doesn't exist
        ValueError: If target_fs is invalid
        
    Example:
        >>> signals, meta = preprocess_edf_signals(
        ...     Path("data/patient001.edf"), 
        ...     target_fs=250
        ... )
    """
```

### Testing Requirements
- Write unit tests for new functions
- Maintain test coverage above 80%
- Use pytest for testing framework
- Mock external dependencies (file I/O, network calls)

**Run tests**:
```bash
pytest tests/ -v --cov=src/
```

## 🔒 Data and Privacy Guidelines

### Data Handling
- **Never commit sensitive data** to the repository
- Use mock/synthetic data for tests and examples
- Respect the Health Data Nexus Data Use Agreement
- Follow de-identification best practices

### File Restrictions
The following should never be committed:
- `.edf` files (EDF signal data)
- `.csv` files with patient data
- Personal health information (PHI)
- API keys or credentials

## 📝 Pull Request Process

### Before Submitting
1. **Run all checks**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   pytest tests/
   ```

2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** with your changes

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Research Impact
- How does this change affect the research objectives?
- Any impact on model performance or reproducibility?

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data included
```

## 🎓 Research Collaboration

### Academic Standards
- Maintain research integrity and reproducibility
- Document experimental procedures clearly
- Share negative results and failed experiments
- Cite relevant prior work and acknowledge contributions

### Communication
- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for research questions
- Contact maintainers for sensitive research matters
- Join project meetings if invited as collaborator

## 📊 Experiment Tracking

### Model Experiments
- Use consistent experiment naming conventions
- Document hyperparameters and configuration
- Save reproducible seeds and environment info
- Track metrics with tensorboard or wandb

### Results Sharing
- Include statistical significance testing
- Provide confidence intervals where appropriate
- Share code for reproducing figures and tables
- Document computational requirements

## 🚀 Release Process

### Version Numbering
Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Research Milestones
- **v0.x.x**: Experimental/development versions
- **v1.0.0**: First stable research release
- **v1.x.x**: Incremental improvements and findings

## 📞 Getting Help

### Questions and Support
- **GitHub Issues**: Technical bugs and feature requests
- **GitHub Discussions**: Research questions and methodology
- **Email**: Principal investigators for sensitive matters

### Research Ethics
For questions about research ethics, data use, or compliance:
- Contact the principal investigator
- Refer to Sunnybrook REB #6197 documentation
- Review T-CAIREM guidelines

## 🏆 Recognition

### Contributor Attribution
- All contributors will be acknowledged in publications
- Significant contributors may be offered co-authorship
- Contributions tracked in CONTRIBUTORS.md file

### Publication Policy
- Contributors should discuss publication plans with PIs
- Follow academic authorship guidelines
- Respect intellectual property agreements

---

Thank you for contributing to advancing sleep-cardiac research! 🚀

**Questions?** Contact the project maintainers or open a GitHub Discussion.
