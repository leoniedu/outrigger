# GitHub Setup Instructions

## Prerequisites

1. GitHub account (create one at https://github.com)
2. Git installed on your local machine

## Steps to Push to GitHub

### 1. Create a New Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `outrigger-rotation-analyzer`
3. Description: "R tool for optimizing crew rotation in long-distance outrigger canoe races"
4. **Important**: Do NOT initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### 2. Copy the Project to Your Local Machine

Download or copy the entire `outrigger-rotation-analyzer` directory to your local machine.

### 3. Link Local Repository to GitHub

In your terminal, navigate to the project directory and run:

```bash
cd outrigger-rotation-analyzer

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/outrigger-rotation-analyzer.git

# Rename default branch to main (optional but recommended)
git branch -M main

# Push to GitHub
git push -u origin main
```

### 4. Enter Your Credentials

When prompted, enter your GitHub credentials:
- **Username**: Your GitHub username
- **Password**: Your Personal Access Token (not your account password)

#### Creating a Personal Access Token:

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name: "Outrigger Analyzer"
4. Select scopes: Check `repo` (all repository permissions)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)
7. Use this token as your password when pushing

### 5. Verify Upload

Visit `https://github.com/YOUR_USERNAME/outrigger-rotation-analyzer` to see your repository.

## Alternative: Using SSH

If you prefer SSH over HTTPS:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key to GitHub
# Copy the contents of ~/.ssh/id_ed25519.pub
# Go to GitHub Settings → SSH and GPG keys → New SSH key
# Paste and save

# Use SSH remote instead
git remote add origin git@github.com:YOUR_USERNAME/outrigger-rotation-analyzer.git
git branch -M main
git push -u origin main
```

## Making Future Updates

After initial setup, to push changes:

```bash
# Make your changes to files

# Stage changes
git add .

# Commit with a message
git commit -m "Add feature: custom stint patterns"

# Push to GitHub
git push
```

## Repository Settings (Optional)

After pushing, you can configure:

### Add Topics (for discoverability)
Go to repository → About (gear icon) → Add topics:
- `outrigger`
- `canoe-racing`
- `sports-analytics`
- `r-programming`
- `crew-rotation`

### Add Description
"R tool for optimizing crew rotation strategies in long-distance outrigger canoe races (OC6)"

### Website
If you deploy documentation: Add the URL

## Troubleshooting

**Error: "remote origin already exists"**
```bash
git remote remove origin
# Then try adding origin again
```

**Error: "failed to push some refs"**
```bash
git pull origin main --rebase
git push -u origin main
```

**Permission denied**
- Check your Personal Access Token has `repo` permissions
- Verify you're using the correct username

## Next Steps

After successful push:
- ✅ Add a nice repository banner/logo
- ✅ Enable GitHub Pages for documentation (optional)
- ✅ Set up GitHub Actions for testing (optional)
- ✅ Invite collaborators if working with team
- ✅ Star your repository!

## Example Repository URL

After setup, your repository will be at:
```
https://github.com/YOUR_USERNAME/outrigger-rotation-analyzer
```

Share this with your crew!
