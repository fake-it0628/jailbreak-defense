# Contributing to HiSCaM

Thank you for your interest in contributing to HiSCaM! We welcome all contributions that help improve LLM safety and jailbreak defense.

## How to Contribute

### 1. Reporting Bugs
- Search existing issues to see if the bug has already been reported.
- If not, open a new issue with a clear description, reproduction steps, and expected vs actual behavior.

### 2. Suggesting Enhancements
- Open an issue to discuss your proposed enhancement.
- Provide a clear rationale for why this feature would be useful.

### 3. Pull Requests
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Run tests to ensure everything is working (`pytest`).
5. Format your code (`black src tests`).
6. Commit your changes with clear, descriptive messages.
7. Push to your branch and open a Pull Request.

## Development Setup

```bash
# Clone and install in editable mode
git clone https://github.com/fake-it0628/jailbreak-defense.git
cd jailbreak-defense
pip install -e .[dev]
```

## Code Style
We use `black` for formatting and `isort` for import sorting. Please ensure your code follows these standards before submitting a PR.

## License
By contributing to HiSCaM, you agree that your contributions will be licensed under its MIT License.
