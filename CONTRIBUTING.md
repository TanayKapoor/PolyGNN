# Contributing to PolyGNN

Thank you for your interest in contributing to PolyGNN! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/yourusername/PolyGNN.git`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Create a branch**: `git checkout -b feature/your-feature-name`
5. **Make changes and commit**: `git commit -m "feat: your feature description"`
6. **Push and create PR**: `git push origin feature/your-feature-name`

## 🛠️ Development Setup

### Environment Setup
```bash
# Create conda environment
conda create -n polygnn python=3.8
conda activate polygnn

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development tools
pip install black flake8 pytest isort
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src/
```

### Code Quality
```bash
# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/
```

## 📝 Contribution Guidelines

### Code Style
- Use **Black** for code formatting (88 character line length)
- Use **isort** for import sorting
- Follow **PEP 8** naming conventions
- Add **type hints** where appropriate
- Write **comprehensive docstrings** (Google style)

### Example Function
```python
def calculate_polymer_features(smiles: str, radius: int = 2) -> Dict[str, float]:
    """Calculate comprehensive polymer features from SMILES.
    
    Args:
        smiles: SMILES string representation of polymer
        radius: Morgan fingerprint radius
        
    Returns:
        Dictionary containing calculated features
        
    Raises:
        ValueError: If SMILES is invalid
    """
    # Implementation here
    pass
```

### Commit Messages
Use conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation changes
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `style:` - Code style changes

Example:
```
feat: add SHAP feature importance analysis

- Implement SHAP-based feature importance
- Add visualization for top polymer features  
- Include statistical significance testing
```

### Testing Requirements
- **All new features** must include tests
- **Maintain >80% test coverage**
- **Test both success and failure cases**
- **Include integration tests** for new components

### Documentation Requirements
- **Update README.md** for new features
- **Add docstrings** to all functions/classes
- **Include usage examples** in docstrings
- **Update relevant documentation** in `docs/`

## 🧪 Areas for Contribution

### High Priority
- **Additional GNN architectures** (GAT, GraphSAGE, etc.)
- **Multi-task learning** improvements
- **Feature engineering** enhancements
- **Uncertainty quantification** methods

### Medium Priority  
- **Performance optimizations**
- **Additional polymer properties** (Tm, density, etc.)
- **Visualization improvements**
- **Documentation enhancements**

### Research Areas
- **New polymer descriptors**
- **Alternative uncertainty methods**
- **Transfer learning** approaches
- **Active learning** for data collection

## 🐛 Bug Reports

When reporting bugs, please include:
- **Python version** and OS
- **Complete error traceback**
- **Minimal code to reproduce**
- **Expected vs actual behavior**
- **Package versions** (`pip freeze`)

## 💡 Feature Requests

For feature requests, please provide:
- **Clear description** of the feature
- **Use case** and motivation
- **Proposed implementation** approach
- **Potential impact** on existing code

## 📚 Documentation

### Adding New Documentation
- Place in appropriate `docs/` subdirectory
- Use **Markdown** format
- Include **code examples**
- **Link from main README.md**

### Code Documentation
- **Docstrings** for all public functions
- **Inline comments** for complex logic
- **Type hints** where appropriate
- **Usage examples** in docstrings

## 🚦 Pull Request Process

### Before Submitting
1. **Run all tests**: `pytest tests/`
2. **Check code style**: `black --check src/ && flake8 src/`
3. **Update documentation** if needed
4. **Add tests** for new features
5. **Verify no breaking changes**

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new features
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process
1. **Automated checks** must pass
2. **At least one review** required
3. **All discussions** must be resolved
4. **Squash and merge** preferred

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** acknowledgments
- **Release notes** for significant contributions
- **Author lists** for research publications

## 📞 Getting Help

- **Open an issue** for bugs/features
- **Start a discussion** for questions
- **Check existing issues** before creating new ones
- **Join community discussions** in issues/PRs

## 📄 License

By contributing, you agree that your contributions will be licensed under the same **MIT License** that covers the project.

---

Thank you for contributing to PolyGNN! 🚀🧪