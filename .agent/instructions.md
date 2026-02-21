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

## Code Formatting
- **Mandatory after every edit**: run `clang-format -i --style=file` on each
	modified `.cu`, `.cpp`, `.cuh`, `.h`, and `.hpp` file before responding.
- This is a required per-file step and must not be skipped.
- Do **not** use `make format` as a substitute for this requirement.

## Development Environment
- Code can not be built or tested locally. All building and testing must be done on Perlmutter.
- Use `make` targets for routine tasks (remote-sync, remote-build, remote-test, etc.)
- Use `fetch_webpage` for external documentation links.
