#!/usr/bin/env python3
"""
Git LFS Uploader Script
Automates the process of setting up Git LFS and uploading large files to GitHub.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


class GitLFSUploader:
    def __init__(self, repo_path=None):
        """Initialize the Git LFS uploader."""
        self.repo_path = repo_path or os.getcwd()
        self.git_attributes_path = os.path.join(self.repo_path, '.gitattributes')
        
    def run_command(self, command, cwd=None, check=True):
        """Run a shell command and return the result."""
        cwd = cwd or self.repo_path
        print(f"Running: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"Error output: {e.stderr}")
            if check:
                raise
            return e
    
    def check_git_lfs_installed(self):
        """Check if Git LFS is installed."""
        print("Checking if Git LFS is installed...")
        result = self.run_command("git lfs version", check=False)
        if result.returncode != 0:
            print("Git LFS is not installed. Please install it first:")
            print("Windows: https://git-lfs.github.com/")
            print("Or run: winget install GitHub.GitLFS")
            return False
        print("Git LFS is installed!")
        return True
    
    def check_git_repo(self):
        """Check if current directory is a Git repository."""
        print("Checking if this is a Git repository...")
        result = self.run_command("git status", check=False)
        if result.returncode != 0:
            print("This is not a Git repository. Please initialize it first:")
            print("git init")
            print("git remote add origin <your-github-repo-url>")
            return False
        print("Git repository found!")
        return True
    
    def setup_git_lfs(self):
        """Initialize Git LFS in the repository."""
        print("Setting up Git LFS...")
        self.run_command("git lfs install")
        print("Git LFS initialized!")
    
    def track_file_types(self, file_patterns):
        """Track specific file patterns with Git LFS."""
        print(f"Tracking file patterns: {file_patterns}")
        for pattern in file_patterns:
            self.run_command(f'git lfs track "{pattern}"')
        print("File patterns tracked!")
    
    def add_files(self, file_paths):
        """Add files to the repository."""
        print(f"Adding files: {file_paths}")
        for file_path in file_paths:
            if os.path.exists(file_path):
                self.run_command(f'git add "{file_path}"')
                print(f"Added: {file_path}")
            else:
                print(f"Warning: File not found: {file_path}")
    
    def add_gitattributes(self):
        """Add .gitattributes file to the repository."""
        print("Adding .gitattributes file...")
        self.run_command("git add .gitattributes")
        print(".gitattributes file added!")
    
    def commit_changes(self, commit_message):
        """Commit changes to the repository."""
        print(f"Committing changes: {commit_message}")
        self.run_command(f'git commit -m "{commit_message}"')
        print("Changes committed!")
    
    def push_to_github(self):
        """Push changes to GitHub."""
        print("Pushing to GitHub...")
        self.run_command("git push")
        print("Successfully pushed to GitHub!")
    
    def upload_large_files(self, file_patterns=None, specific_files=None, commit_message="Add large files via Git LFS"):
        """Main method to upload large files using Git LFS."""
        if file_patterns is None:
            file_patterns = ["*.mbox", "*.zip", "*.tar.gz", "*.7z"]
        
        print("=== Git LFS Uploader ===")
        print(f"Repository path: {self.repo_path}")
        
        # Check prerequisites
        if not self.check_git_lfs_installed():
            return False
        
        if not self.check_git_repo():
            return False
        
        # Setup Git LFS
        self.setup_git_lfs()
        
        # Track file patterns
        self.track_file_types(file_patterns)
        
        # Add .gitattributes file
        self.add_gitattributes()
        
        # Add specific files if provided
        if specific_files:
            self.add_files(specific_files)
        
        # Commit changes
        self.commit_changes(commit_message)
        
        # Push to GitHub
        self.push_to_github()
        
        print("=== Upload Complete ===")
        return True


def main():
    """Main function to run the Git LFS uploader."""
    # Define the large files to upload
    large_files = [
        "extracted/Takeout/Mail/All mail Including Spam and Trash.mbox"
    ]
    
    # File patterns to track with Git LFS
    file_patterns = [
        "*.mbox",      # Mailbox files
        "*.zip",       # Archive files
        "*.tar.gz",    # Compressed archives
        "*.7z",        # 7-Zip archives
        "*.psd",       # Photoshop files (as mentioned in your example)
        "*.ai",        # Illustrator files
        "*.sketch",    # Sketch files
        "*.mov",       # Video files
        "*.mp4",
        "*.avi",
        "*.wmv"
    ]
    
    # Create uploader instance
    uploader = GitLFSUploader()
    
    # Upload the files
    success = uploader.upload_large_files(
        file_patterns=file_patterns,
        specific_files=large_files,
        commit_message="Add email data via Git LFS"
    )
    
    if success:
        print("\n✅ Successfully uploaded large files to GitHub using Git LFS!")
        print("You can now view your files on GitHub.")
    else:
        print("\n❌ Failed to upload files. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 