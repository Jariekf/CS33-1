# GitHub Commit Guide v2.0

Update: Yiting Liu Author: Liam (Zijian Xue), 2026; based on the "github-clone-and-push.pdf"

This guide explains how to set up Git, clone the course repository, create your own development branch, commit changes, push to GitHub, and submit a pull request (PR) correctly for the team project.

---

## 1. Install Git

Before working with the project, make sure Git is installed on your computer.

### macOS

Install Git with Homebrew:

```bash
brew install git
git --version
```

### Windows

Download and install **Git for Windows**:

`https://git-scm.com/download/win`

During installation, the default options are usually fine. After installation, open **Git Bash**, **PowerShell**, or **Command Prompt** and verify:

```bash
git --version
```

## 2. Clone the Repository

Use HTTPS to clone the team repository to your local machine:

```bash
git clone https://github.sydney.edu.au/CS33-1/CS33-1.git
cd CS33-1
```


- On macOS, run the commands in Terminal.
- On Windows, use Git Bash, PowerShell, or Command Prompt.

For login:

- Use your **USyd GitHub Enterprise** account.
- Depending on the university configuration, Git may ask for your **UniKey password** or a **Personal Access Token (PAT)** instead of a password.

## 3. Create Your Personal Development Branch

Do not work directly on `main`. Create your own development branch before starting a task.

Recommended branch naming format:

```text
name_develop
```

Example:

```text
jerry_develop
```

Create the branch from the repository root:

```bash
git checkout main
git pull origin main
git checkout -b jerry_develop
```

You can confirm your current branch with:

```bash
git branch
```

The active branch will be marked with `*`.

## 4. Push Your Branch to Remote for the First Time

The first time you push your branch, use `-u` to set the upstream tracking relationship:

```bash
git push -u origin jerry_develop
```

## 5. Work on Your Branch

Make all your changes on your own branch.

Use clear commit messages in this format:

[Module/Task] Short description

Example:

```bash
git add .
git commit -m "[Login] Implement Manager agent"
git push
```

 All commit messages are recommended to follow this format:

```text
- Recomnebded format: <type>: <brief description>.

- Types:

feat: A new feature
fix: A bug fix
docs: Documentation changes
style: Code style changes (non-functional)
refactor: Code refactoring (neither adding features nor fixing bugs)
test: Adding tests
chore: Changes to the build process or auxiliary tools
```
Examples: feat: Add Manager Agent | fix: Fix Supervisor Agent reporting issues


## 6. Submit a Pull Request (PR) --> Merge PR into `main`

On GitHub, go to your branch → click Compare & pull request.

Fill in the PR title (same as commit message is fine).

Add a description if you like.

If your code requires review by others, please add the members who need to review it.

Click Create pull request.

❗ Do not push directly to main. All changes must go through a PR.

After the PR is approved, you can click merge.

## 8. Sync Your Branch Before Starting New Work

Sync Your Development Branch ❗❗❗
### Very important thing!!!

Before starting any task, make sure your branch is up-to-date:

```bash
git checkout develop-alice        # Switch to your development branch
git fetch origin                  # Fetch latest changes from remote
git merge origin/main             # Merge latest main into your branch
# OR, if you prefer linear history:
git rebase origin/main
git push origin develop-alice     # Update remote branch
```

## 9. Common Authentication and Push Issues

If Git asks for credentials when cloning or pushing:

- Username: usually your **UniKey** or the username required by USyd GitHub Enterprise
- Password: your UniKey password or a valid **Personal Access Token**, depending on the platform settings

If cloning or pushing still fails with errors such as `401` or `403`: please `@Liam` in the WeChat or Slack group chat

## 10. Quick Command Reference

```bash
git clone https://github.sydney.edu.au/CS33-1/CS33-1.git  # Clone the repository to your local machine
cd CS33-1                                                 # Enter the repository root directory
git checkout -b name_develop                              # Create and switch to your personal branch
git push -u origin name_develop                           # Push the branch for the first time and set upstream
git add .                                                 # Stage your changes
git commit -m "feat: Short description"                   # Commit with the recommended message format
git push                                                  # Push your latest commit to remote
```

Replace `name_develop` with your actual branch name.

## 11. Important Notes

- Never push directly to `main`; always use a PR.
- Always work on your own branch.
- Always sync with `main` before starting new work.
- Write clear commit messages.
- Make sure your code works and does not contain unresolved conflicts before opening or merging a PR.
