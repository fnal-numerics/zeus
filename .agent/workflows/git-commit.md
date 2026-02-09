---
description: Create a git commit message following repository standards
---

# Workflow: Git Commit Message

Follow these steps to generate or refine a git commit message:

1. **Analyze Changes**: Compare current changes against HEAD using `git diff` or workspace tools.
2. **Subject Line**:
   - Limit to 40 characters.
   - Use imperative mood ("Add feature", not "Added feature").
   - Capitalize first letter.
   - No period at the end.
3. **Body**:
   - Insert a blank line after the subject.
   - Wrap lines at 72 characters.
   - Focus on *what* and *why*.
   - Mention all non-formatting changes.
4. **Reformat Only**:
   - If changes are only whitespace/line breaks, use exactly: `Reformat code`

// turbo
## Example Usage
To get a commit message suggestion based on staged changes:
1. Run `git diff --cached` to see what is ready to be committed.
2. Use this workflow to draft the message.
