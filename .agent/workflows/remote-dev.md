---
description: Sync, build, and test the project on a remote machine (Perlmutter)
---

This workflow allows you to automate the process of syncing local changes to Perlmutter, building the code, and running tests.

### Steps

1. **Sync changes to Perlmutter**
// turbo
```bash
make remote-sync
```

2. **Build on Perlmutter**
// turbo
```bash
make remote-build
```

3. **Run tests on Perlmutter**
// turbo
```bash
make remote-test
```

### Usage Tips
- Ensure your SSH agent is running and has your NERSC key loaded.
- You can run these steps individually or as a complete cycle.
- The remote project directory is configured in `.remote_config`.
