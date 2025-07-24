#!/usr/bin/env python3
"""
Simple Git LFS Uploader
A straightforward script to upload large files to GitHub using Git LFS.
"""

import os
import subprocess
import sys


def run_command(command, check=True):
    """Run a shell command and print output."""
    print(f"🔄 Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        if check:
            raise
        return e


def upload_to_git_lfs():
    """Upload large files to GitHub using Git LFS."""
    
    print("🚀 Starting Git LFS Upload Process...")
    
    # Step 1: Check if Git LFS is installed
    print("\n📋 Step 1: Checking Git LFS installation...")
    result = run_command("git lfs version", check=False)
    if result.returncode != 0:
        print("❌ Git LFS is not installed!")
        print("💡 Install it from: https://git-lfs.github.com/")
        print("   Or run: winget install GitHub.GitLFS")
        return False
    
    # Step 2: Check if this is a Git repository
    print("\n📋 Step 2: Checking Git repository...")
    result = run_command("git status", check=False)
    if result.returncode != 0:
        print("❌ This is not a Git repository!")
        print("💡 Initialize it with:")
        print("   git init")
        print("   git remote add origin <your-github-repo-url>")
        return False
    
    # Step 3: Setup Git LFS
    print("\n📋 Step 3: Setting up Git LFS...")
    run_command("git lfs install")
    
    # Step 4: Track file patterns
    print("\n📋 Step 4: Tracking file patterns with Git LFS...")
    file_patterns = [
        "*.mbox",      # Your email file
        "*.zip",       # Archive files
        "*.tar.gz",    # Compressed archives
        "*.psd",       # Photoshop files
        "*.mov",       # Video files
        "*.mp4",
        "*.avi"
    ]
    
    for pattern in file_patterns:
        run_command(f'git lfs track "{pattern}"')
    
    # Step 5: Add .gitattributes file
    print("\n📋 Step 5: Adding .gitattributes file...")
    run_command("git add .gitattributes")
    
    # Step 6: Add your large file
    print("\n📋 Step 6: Adding large files...")
    large_file = "extracted/Takeout/Mail/All mail Including Spam and Trash.mbox"
    
    if os.path.exists(large_file):
        run_command(f'git add "{large_file}"')
        print(f"✅ Added: {large_file}")
    else:
        print(f"⚠️  Warning: File not found: {large_file}")
        print("💡 Make sure the file path is correct!")
    
    # Step 7: Commit changes
    print("\n📋 Step 7: Committing changes...")
    run_command('git commit -m "Add large files via Git LFS"')
    
    # Step 8: Push to GitHub
    print("\n📋 Step 8: Pushing to GitHub...")
    run_command("git push")
    
    print("\n🎉 Success! Your large files have been uploaded to GitHub using Git LFS!")
    print("📁 You can now view them on your GitHub repository.")
    return True


if __name__ == "__main__":
    try:
        success = upload_to_git_lfs()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1) 