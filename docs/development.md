# GraSP Development Guide

This guide covers development practices for the GraSP project, including code style, linting, and testing procedures.

## Development Setup

To set up your development environment:

```bash
  # Set up development environment with all dev tools
  make setup-dev
```

## Code Style and Linting

GraSP uses several tools to ensure code quality:

- **Black**: For code formatting
- **isort**: For import sorting
- **ruff**: For code quality analysis
- **mypy**: For static type checking

### Using the Linting Tools

You can run all linters at once with:

```bash
  make lint
```

Or run individual linters:

```bash
  make lint-ruff  # Run ruff
  make lint-mypy    # Run mypy
```

### Code Formatting

To format your code according to the project standards:

```bash
  make format
```

This will run both black and isort. You can run them individually:

```bash
  make format-black  # Format code with black
  make format-isort  # Sort imports with isort
```

### Checking Format Without Modifying Files

If you want to check your code formatting without changing files:

```bash
  make check-format
```

Or check specific formatters:

```bash
  make check-format-black
  make check-format-isort
```

## Testing

Run the test suite:

```bash
  make test
```

Run tests with verbose output:

```bash
  make test-verbose
```

Run tests with coverage:

```bash
  make test-coverage
```

## Continuous Integration

Run all CI steps locally:

```bash
  make ci
```

This runs formatting, linting, and tests in sequence.

## Release Process

1. Update version numbers in `pyproject.toml`
2. Update CHANGELOG.md
3. Run tests, formatting and linting: 
    ```bash 
    make ci
    ```
4. Build the package: 
    ```bash
    make build
    ```
5. Push changes and create a new GitHub release

## Configuration

Configuration for all tools is in `pyproject.toml`, including:

- Black configuration: `[tool.black]` section
- isort configuration: `[tool.isort]` section
- ruff configuration: `[tool.ruff]` section
- mypy configuration: `[tool.mypy]` section

You can customize these settings to match your project's specific requirements.