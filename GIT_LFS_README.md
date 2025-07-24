# Git LFS Uploader Scripts

This repository contains Python scripts to automate the process of uploading large files to GitHub using Git Large File Storage (LFS).

## What is Git LFS?

Git LFS (Large File Storage) is a Git extension that allows you to store large files on GitHub without bloating your repository. Instead of storing the actual large files in your Git history, Git LFS stores pointers to the files and uploads the actual content to a separate storage system.

## Files Included

1. **`git_lfs_uploader.py`** - Comprehensive Git LFS uploader with class-based structure
2. **`simple_git_lfs_uploader.py`** - Simple, straightforward script for quick uploads
3. **`GIT_LFS_README.md`** - This documentation file

## Prerequisites

Before running the scripts, make sure you have:

1. **Git LFS installed** on your system
   - Windows: Download from https://git-lfs.github.com/
   - Or run: `winget install GitHub.GitLFS`

2. **Git repository initialized** and connected to GitHub
   ```bash
   git init
   git remote add origin <your-github-repo-url>
   ```

3. **Python 3.6+** installed on your system

## Quick Start

### Option 1: Simple Script (Recommended for beginners)

```bash
python simple_git_lfs_uploader.py
```

This script will:
- Check if Git LFS is installed
- Verify you're in a Git repository
- Set up Git LFS tracking for common large file types
- Upload your `.mbox` file (135MB email data)
- Push everything to GitHub

### Option 2: Advanced Script

```bash
python git_lfs_uploader.py
```

This script provides more customization options and better error handling.

## What the Scripts Do

The scripts automate these Git LFS commands:

1. **`git lfs install`** - Initialize Git LFS in your repository
2. **`git lfs track "*.mbox"`** - Tell Git LFS to track `.mbox` files
3. **`git add .gitattributes`** - Add the tracking configuration
4. **`git add "extracted/Takeout/Mail/All mail Including Spam and Trash.mbox"`** - Add your large file
5. **`git commit -m "Add large files via Git LFS"`** - Commit the changes
6. **`git push`** - Push to GitHub

## File Types Tracked

The scripts automatically track these file types with Git LFS:

- `*.mbox` - Mailbox files (your email data)
- `*.zip` - Archive files
- `*.tar.gz` - Compressed archives
- `*.psd` - Photoshop files
- `*.mov`, `*.mp4`, `*.avi` - Video files
- `*.ai` - Illustrator files
- `*.sketch` - Sketch files
- `*.7z` - 7-Zip archives

## Customization

### Adding More File Types

Edit the `file_patterns` list in either script:

```python
file_patterns = [
    "*.mbox",
    "*.zip",
    "*.your-custom-extension"  # Add your own
]
```

### Uploading Different Files

Edit the `large_files` list in the advanced script:

```python
large_files = [
    "path/to/your/large/file.ext",
    "another/large/file.ext"
]
```

## Troubleshooting

### Git LFS Not Installed
```
❌ Git LFS is not installed!
💡 Install it from: https://git-lfs.github.com/
```

**Solution**: Download and install Git LFS from the official website.

### Not a Git Repository
```
❌ This is not a Git repository!
💡 Initialize it with:
   git init
   git remote add origin <your-github-repo-url>
```

**Solution**: Initialize your Git repository and connect it to GitHub.

### File Not Found
```
⚠️ Warning: File not found: extracted/Takeout/Mail/All mail Including Spam and Trash.mbox
```

**Solution**: Check that the file path is correct and the file exists.

### Push Permission Denied
```
❌ Error: Permission denied
```

**Solution**: Make sure you have write access to the GitHub repository and your Git credentials are configured.

## Manual Git LFS Commands

If you prefer to run the commands manually:

```bash
# 1. Install Git LFS (if not already installed)
# Download from https://git-lfs.github.com/

# 2. Initialize Git LFS
git lfs install

# 3. Track file types
git lfs track "*.mbox"
git lfs track "*.zip"
git lfs track "*.psd"

# 4. Add .gitattributes file
git add .gitattributes

# 5. Add your large file
git add "extracted/Takeout/Mail/All mail Including Spam and Trash.mbox"

# 6. Commit
git commit -m "Add large files via Git LFS"

# 7. Push to GitHub
git push
```

## Benefits of Using Git LFS

1. **Faster clones** - Large files are downloaded only when needed
2. **Smaller repository size** - Git history doesn't include large file content
3. **Better performance** - Git operations are faster
4. **Version control** - Still track changes to large files
5. **GitHub integration** - Seamless integration with GitHub's interface

## File Size Limits

- **GitHub LFS**: 2GB per file (free tier)
- **GitHub LFS Bandwidth**: 1GB/month (free tier)
- **GitHub LFS Storage**: 1GB (free tier)

For larger files or higher limits, consider GitHub Pro or GitHub Enterprise.

## Support

If you encounter issues:

1. Check that Git LFS is properly installed
2. Verify your Git repository is connected to GitHub
3. Ensure you have the correct file paths
4. Check your GitHub repository permissions

---

**Note**: These scripts are designed to work with the specific file structure in your email analysis project. Adjust file paths as needed for your specific use case. 