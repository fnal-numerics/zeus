# Project Instructions

## Project Status
This repository is currently focused on C++/CUDA development with remote building and testing on Perlmutter.

## Interaction & Safety
To activate these project rules globally, copy the following sections into your global Antigravity `GEMINI.md` or "User Rules" settings:

### Communication Style
- Act as professional colleague; avoid sycophancy; point out mistakes constructively.
- Say "I don't know" when lacking information; never fabricate answers.
- Use literal language; state assumptions explicitly; ask clarifying questions.

### Repository Management
- **Strict Index Boundary**: Treat the git "staged" area (index) as a read-only boundary during analysis tasks. Do not run `git add` unless specifically requested.
- **Explicit Command Requirement**: Never run `git add`, `git commit`, or `git push` without an explicit user command for that specific action.

## Development Environment
- Do not create new virtual environments; rely on the active conda environment.
- Use `make` targets for routine tasks (format, lint, test).
- Use `fetch_webpage` for external documentation links.
