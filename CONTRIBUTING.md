# Contributing to GenAI RAG Boilerplate

Thank you for your interest in contributing to the GenAI RAG Boilerplate! We welcome contributions from the community and are grateful for your support.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git
- A GitHub account

### Development Setup

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/your-username/genai-boilerplate-milvus-python.git
   cd genai-boilerplate-milvus-python
   ```

2. **Set up the development environment**
   ```bash
   # Run the setup script
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   
   # Or manually:
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Run tests**
   ```bash
   cd backend
   python examples/api_test.py
   python examples/rag_example.py
   ```

## 🛠️ Development Guidelines

### Code Style

- **Python**: Follow PEP 8, use Black for formatting
- **Type Hints**: Use type hints for all functions and methods
- **Docstrings**: Use Google-style docstrings
- **Comments**: Write clear, concise comments for complex logic

### Code Quality

```bash
# Format code
black backend/src/

# Check linting
ruff check backend/src/

# Type checking
mypy backend/src/
```

### Project Structure

```
backend/src/
├── api/           # FastAPI routes and endpoints
├── core/          # Core functionality (database, clients)
├── models/        # Pydantic models and schemas
├── services/      # Business logic and services
└── utils/         # Utility functions and helpers
```

## 📝 How to Contribute

### Reporting Issues

When reporting issues, please include:

- **Environment details**: OS, Python version, Docker version
- **Steps to reproduce**: Clear, step-by-step instructions
- **Expected vs actual behavior**
- **Error messages and logs**
- **Screenshots** (if applicable)

### Suggesting Features

For feature requests:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: How you envision the feature working
- **Alternatives considered**: Other approaches you've thought about
- **Implementation ideas**: Technical approach (if applicable)

### Submitting Pull Requests

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description
   
   - Detailed description of changes
   - Any breaking changes noted
   - Closes #issue-number"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

### Commit Message Format

We use conventional commits:

```
type(scope): brief description

Detailed description of changes (if needed)

- List specific changes
- Include breaking changes
- Reference issues: Closes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring (no feature changes)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🧪 Testing

### Running Tests

```bash
# API integration tests
python backend/examples/api_test.py

# RAG functionality tests
python backend/examples/rag_example.py

# Unit tests (when available)
pytest backend/tests/
```

### Writing Tests

- Write tests for new features and bug fixes
- Use descriptive test names
- Include both positive and negative test cases
- Test edge cases and error conditions

### Test Categories

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test REST endpoints
4. **End-to-End Tests**: Test complete workflows

## 📚 Documentation

### Documentation Updates

- Update README.md for new features
- Add docstrings to new functions/classes
- Update API documentation
- Include usage examples

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams for complex concepts
- Keep examples up-to-date

## 🎯 Areas for Contribution

### High Priority

- **Performance Optimization**: Improve query speed and resource usage
- **Error Handling**: Better error messages and recovery mechanisms
- **Testing**: Expand test coverage
- **Documentation**: More examples and tutorials

### Medium Priority

- **New AI Providers**: Integration with additional LLM providers
- **Advanced Features**: Streaming responses, batch processing
- **Security**: Enhanced authentication and authorization
- **Monitoring**: Better observability and metrics

### Low Priority

- **UI Improvements**: Better admin interfaces
- **Integration**: Webhooks, third-party services
- **Optimization**: Code cleanup and refactoring

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different opinions and approaches

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and discussions
- **Discussions**: General questions and community interaction

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review for quality and style
3. **Testing**: Manual testing for complex features
4. **Documentation**: Ensure docs are updated
5. **Merge**: Approved PRs are merged to main

## 🏷️ Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Steps

1. Update version numbers
2. Update CHANGELOG.md
3. Create release branch
4. Tag release
5. Update documentation
6. Announce release

## ❓ Getting Help

### Resources

- **Documentation**: README.md and inline docs
- **Examples**: Complete examples in `backend/examples/`
- **Issues**: Search existing issues first
- **Discussions**: Community Q&A

### Contact

- **GitHub Issues**: Technical problems
- **GitHub Discussions**: Questions and ideas
- **Email**: For security issues only

## 🎉 Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **Release Notes**: Notable contributions
- **GitHub**: Contributor badges

Thank you for contributing to making GenAI RAG Boilerplate better for everyone! 🚀
