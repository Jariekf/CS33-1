# Mac / Windows：Git 远程仓库拉取与上传（Usyd GitHub）

本文说明在 **macOS** 与 **Windows** 上安装 Git、克隆课程仓库、按规范创建分支并推送到远程的步骤。远程地址（HTTPS）：[https://github.sydney.edu.au/CS33-1/CS33-1.git](https://github.sydney.edu.au/CS33-1/CS33-1.git)。

---

## 1. 安装 Git

### macOS

使用 [Homebrew](https://brew.sh/) 安装：

```bash
brew install git
```

验证：

```bash
git --version
```

### Windows

下载并安装官方 [Git for Windows](https://git-scm.com/download/win)，安装时保持默认选项即可（会安装 **Git Bash**，便于使用与 Mac 类似的命令）。
安装完成后打开 **Git Bash** 或 **PowerShell**，验证：
  ```bash
  git --version
  ```

---

## 2. 拉取仓库到本地（克隆）

使用 **HTTPS** 克隆。首次 `clone` / `push` 时，Git 会提示登录：请使用你在 [USyd GitHub Enterprise](https://github.sydney.edu.au) 的账号（一般为 **UniKey + LDAP 登录**，与网页端一致；若学校要求使用 **Personal Access Token** 代替密码，以课程或 [Sydney Informatics Hub 说明](https://github.sydney.edu.au) 为准）。

在你要存放项目的目录下执行：

```bash
git clone https://github.sydney.edu.au/CS33-1/CS33-1.git
cd CS33-1
```

- **Mac**：在「终端」中执行上述命令。  
- **Windows**：在 **Git Bash/cmd/powershell** 中执行（路径建议不要含中文或特殊字符，避免编码问题）。

克隆成功后，当前目录下会有 `CS33-1` 文件夹，且默认在远程跟踪的默认分支（多为 `main`）上。

---

## 3. 在 `CS33-1` 中创建自己的分支

分支命名规则：**`name_develop`**（将 `name` 换成你的名字或学号等课程要求的一致标识，例如'zijian_develop'）。

确保已在仓库根目录：

```bash
cd CS33-1
```

创建并切换到新分支：

```bash
git checkout -b name_develop
```

将 `name_develop` 换成你的实际分支名。可用下面命令确认当前分支：

```bash
git branch
```

当前分支前会有 `*`。

---

## 4. 将本地分支推送到远程

首次把该分支推到远程并建立跟踪关系：

```bash
git push -u origin name_develop
```

同样把 `name_develop` 换成你的分支名。

之后在同一分支上若已有上游跟踪，日常推送可用：

```bash
git add .
git commit -m "你的提交说明"
git push
```

这里的提交说明在群文件'Git Usage Guide'中有详细的规则

---

## 补充：HTTPS 登录常见问题

**用户名**：多为你的 **UniKey**（或网页登录时使用的用户名，以实际提示为准）。  
**密码**：可能是 UniKey 密码，或学校要求的 **Token**（在 GitHub Enterprise 网页上生成 Personal Access Token 后，在密码处粘贴 Token）。  
**报错**：若克隆或推送仍报 401/403,请在群里@liam
---

## 命令速查

| 步骤     | 命令 |
|----------|------|
| 克隆     | `git clone https://github.sydney.edu.au/CS33-1/CS33-1.git` |
| 进入目录 | `cd CS33-1` |
| 新建分支 | `git checkout -b name_develop` |
| 首次推送 | `git push -u origin name_develop` |

将文中所有 `name_develop` 替换为你自己的分支名即可。
