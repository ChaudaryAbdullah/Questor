# Contributing to Questor

Thank you for your interest in contributing to Questor! 🎉

## How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include detailed steps to reproduce
- Provide system information (OS, Python version)
- Include relevant logs or screenshots

### Suggesting Features
- Open an issue with the `enhancement` label
- Describe the feature and its benefits
- Provide use cases and examples

### Code Contributions

#### 1. Fork and Clone
```bash
git clone https://github.com/YourUsername/Questor.git
cd Questor
```

#### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

#### 3. Make Changes
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Keep commits focused and atomic

#### 4. Test Your Changes
```bash
# Run tests
pytest tests/

# Check code style
black Pipelines/
isort Pipelines/

# Type checking
mypy Pipelines/
```

#### 5. Submit a Pull Request
- Write a clear PR description
- Reference related issues
- Ensure CI passes

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt
```

## Code Style

- Use `black` for formatting
- Use `isort` for import sorting
- Follow type hints where possible
- Write docstrings for all public functions

## Adding New Agents

1. Create agent in `Pipelines/agents/your_agent.py`
2. Extend `BaseAgent` class
3. Implement required methods
4. Add tests in `tests/agents/`
5. Update `agent_config.yaml`
6. Document in README

## Questions?

Feel free to open an issue or reach out to the maintainers!

---

Thank you for contributing! 🙏
