# Contributing

Thank you for your interest in improving this Deep Learning Lab repository.

## Development Environment
1. Python 3.10+ (recommend 3.10 or 3.11)
2. Create virtual environment:
   ```powershell
   python -m venv .venv ; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Install dev tooling:
   ```powershell
   pip install black isort flake8 pre-commit
   pre-commit install
   ```

## Workflow
- Fork + create feature branch: `feat/<short-description>`
- Keep changes focused (one PR per logical improvement)
- Run `pre-commit` hooks (auto triggered on commit)

## Notebooks
- When adding significant logic, extract reusable code into a Python module (e.g. `ex5/utils.py`) and import into notebooks.
- Avoid very large notebook output cells—clear before commit.

## Adding Dependencies
- Update root `requirements.txt` (alphabetical, unpinned unless reproducibility requires pins).
- Consider impact on other exercises.

## Model & Data Files
- Track large models with Git LFS. Do NOT commit very large raw datasets; prefer download scripts or instructions.

## Commit Messages
Format:
```
<type>: <short summary>

Optional body describing motivation
```
Types: `feat`, `fix`, `docs`, `refactor`, `chore`, `experiment`.

## Pull Request Checklist
- [ ] Feature/bug scope clearly stated
- [ ] No large transient outputs in notebooks
- [ ] Code formatted (black, isort)
- [ ] Lint passes: `flake8 .`
- [ ] README / docs updated if needed

## Issues
Use labels: `bug`, `enhancement`, `question`, `documentation`.

## Roadmap Contributions
See README "Roadmap" section. Propose additions via issue before large changes.

## License
By contributing you agree your contributions are MIT licensed.
