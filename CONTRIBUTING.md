# Contributing to Pokemon Card Generator

Thank you for your interest in contributing to the Pokemon Card Generator! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept criticism gracefully

## Getting Started

### Setting up the development environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/pokemon_card_gen.git
   cd pokemon_card_gen
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development dependencies
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Make your changes**
   - Write clean, readable code
   - Add comments for complex logic
   - Update documentation as needed

2. **Test your changes**
   - Run unit tests: `pytest tests/`
   - Test manually if needed
   - Ensure all tests pass

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Guidelines

We follow conventional commits format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add StyleGAN2 architecture option
fix: resolve memory leak in training loop
docs: update API endpoint documentation
```

## Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces
- **Imports**: Organized using `isort`
- **Formatting**: Use `black` for automatic formatting

Run formatters before committing:
```bash
black .
isort .
flake8 .
```

### Code Quality

- Write docstrings for all functions and classes
- Use type hints where appropriate
- Keep functions focused and concise
- Avoid deep nesting (max 3-4 levels)
- Use meaningful variable names

Example:
```python
def generate_cards(
    generator: nn.Module,
    num_images: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate Pokemon cards using trained generator

    Args:
        generator: Trained generator model
        num_images: Number of cards to generate
        device: Device to generate on

    Returns:
        Generated images tensor
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::TestGenerator::test_generator_forward
```

### Writing Tests

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Test edge cases and error conditions

Example:
```python
def test_generator_output_shape(self):
    """Test generator produces correct output shape"""
    generator = Generator(latent_dim=100)
    noise = torch.randn(4, 100, 1, 1)
    output = generator(noise)

    self.assertEqual(output.shape, (4, 3, 256, 256))
```

## Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Update documentation
   - Add/update tests for your changes
   - Run code formatters

2. **PR Description:**
   - Clearly describe what changes you made
   - Reference related issues
   - Include screenshots if UI changes
   - List any breaking changes

3. **Review Process:**
   - Address reviewer feedback promptly
   - Keep discussions respectful and constructive
   - Make requested changes in new commits

4. **Merging:**
   - PRs require at least one approval
   - All CI checks must pass
   - Squash and merge is preferred

## Reporting Bugs

When reporting bugs, please include:

1. **Clear title and description**
2. **Steps to reproduce**
3. **Expected vs actual behavior**
4. **Environment details:**
   - OS and version
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
5. **Error messages and logs**
6. **Screenshots if applicable**

Use the bug report template when creating issues.

## Suggesting Features

We welcome feature suggestions! Please:

1. **Check existing issues** first
2. **Describe the use case** - why is this needed?
3. **Propose a solution** - how should it work?
4. **Consider alternatives** - are there other ways?
5. **Provide examples** - mockups, code snippets, etc.

Use the feature request template when creating issues.

## Areas for Contribution

We especially welcome contributions in these areas:

### Model Improvements
- Alternative architectures (StyleGAN, ProGAN, etc.)
- Conditional generation (by type, rarity, etc.)
- Model compression and optimization
- Transfer learning approaches

### Training Enhancements
- Advanced augmentation techniques
- Learning rate schedules
- Multi-GPU training
- Mixed precision training improvements

### API & UI
- Advanced filtering and search
- Batch processing queue
- WebSocket support
- Mobile-responsive improvements

### Documentation
- Tutorials and guides
- Architecture diagrams
- Video walkthroughs
- Translation to other languages

### Performance
- Optimization opportunities
- Caching strategies
- Database integration
- Profiling and benchmarking

## Questions?

If you have questions:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with the "question" label
4. Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to Pokemon Card Generator! 🎴
