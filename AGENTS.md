# AI Agents Manifest (AGENTS.md)

This file provides centralized project context and coding standards for all AI agents (Antigravity, GitHub Copilot, Amazon Q).

## Configuration
- **Antigravity**: Follows [.agent/instructions.md](./.agent/instructions.md).

### Setup for Contributors
To ensure Antigravity respects the project's safety boundaries and commit standards, it is **strongly recommended** to add the interaction rules to your global agent settings:

> [!TIP]
> **Antigravity Users**: Copy the contents of the **Repository Management** and **Interaction** sections from [.agent/instructions.md](./.agent/instructions.md) into your global "User Rules" setting. This activates the **Strict Index Boundary** protection.

## Development Workflow
AI agents should use the project `Makefile` for routine tasks:
- `make remote-sync`: Sync files to Perlmutter.
- `make remote-build`: Build on Perlmutter.
- `make remote-test`: Run tests on Perlmutter.
**Do not assume the code can be built or run locally.**
