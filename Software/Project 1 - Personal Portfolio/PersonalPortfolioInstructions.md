# Complete Beginner's Guide: Building Your Personal Website with React and GitHub Pages

Welcome! This tutorial will guide you through creating and publishing your very own personal website using React and GitHub Pages. Don't worry if you've never done this before - we'll walk through every step together.

## What You'll Learn

By the end of this tutorial, you will have:
- A GitHub account and repository
- Visual Studio Code installed and configured
- A React website running on your computer
- Your website published live on the internet at `https://yourusername.github.io`

---

## Part 1: Setting Up GitHub

### Step 1: Create a GitHub Account

1. Go to [github.com](https://github.com) in your web browser
2. Click the **"Sign up"** button in the top right corner
3. Enter your email address and click **"Continue"**
4. Create a password and click **"Continue"**
5. Choose a username (this will be part of your website URL!) and click **"Continue"**
6. Complete the verification puzzle
7. Check your email for a verification code and enter it
8. You can skip the personalization questions or fill them out - either way is fine

**Important:** Remember your username! Your website will be at `https://yourusername.github.io`

### Step 2: Create Your Website Repository

A repository (or "repo") is like a folder that stores all your website files.

1. Once logged into GitHub, click the **"+"** icon in the top right corner
2. Select **"New repository"**
3. In the "Repository name" field, type: `yourusername.github.io`
   - Replace `yourusername` with your actual GitHub username
   - For example, if your username is "johnsmith", type: `johnsmith.github.io`
   - **This exact naming is crucial!** GitHub Pages requires this specific format
4. Add a description (optional): "My personal website"
5. Select **"Public"** (this must be public for GitHub Pages to work with a free account)
6. Check the box that says **"Add a README file"**
7. Click the green **"Create repository"** button

Congratulations! You've created your repository.

### Step 3: Clone Your Repository to Your Computer

"Cloning" means downloading a copy of your repository to work on locally.

1. On your repository page, click the green **"Code"** button
2. Make sure "HTTPS" is selected (it should be underlined)
3. Click the copy icon next to the URL (it looks like two overlapping squares)
4. Open your computer's terminal or command prompt:
   - **Windows:** Press `Win + R`, type `cmd`, and press Enter
   - **Mac:** Press `Cmd + Space`, type "terminal", and press Enter
5. Navigate to where you want to store your project. For example:
   - Type `cd Desktop` and press Enter (this goes to your Desktop)
   - Or type `cd Documents` to go to your Documents folder
6. Type the following command and press Enter:
   ```
   git clone https://github.com/yourusername/yourusername.github.io.git
   ```
   - Replace the URL with the one you copied in step 3
7. You should see text indicating the repository is being cloned
8. Once complete, type `cd yourusername.github.io` to enter your new folder

**Troubleshooting:** If you get an error saying "git is not recognized", you need to install Git first:
- Go to [git-scm.com](https://git-scm.com/downloads)
- Download and install Git for your operating system
- Restart your terminal and try again

---

## Part 2: Installing Your Tools

### Step 4: Download and Install Visual Studio Code

Visual Studio Code (VS Code) is a text editor where you'll write your code.

1. Go to [code.visualstudio.com](https://code.visualstudio.com)
2. Click the big download button for your operating system
3. Once downloaded, run the installer
4. Follow the installation prompts (the default settings are fine)
5. When installation completes, open VS Code

### Step 5: Open Your Repository in VS Code

1. In VS Code, click **"File"** in the top menu
2. Select **"Open Folder"**
3. Navigate to where you cloned your repository (likely Desktop or Documents)
4. Select the `yourusername.github.io` folder
5. Click **"Select Folder"** (or "Open" on Mac)

You should now see your repository files in the left sidebar! Right now it only has a README.md file.

### Step 6: Install Node.js and npm

Node.js is a tool that lets you run JavaScript on your computer, and npm is a package manager that helps install tools like React.

1. Go to [nodejs.org](https://nodejs.org)
2. Download the **LTS (Long Term Support)** version - click the button on the left
3. Run the installer and follow the prompts (default settings are fine)
4. **Important:** Make sure the box for "npm package manager" is checked during installation
5. Restart your computer after installation completes

**Verify Installation:**
1. Open a new terminal in VS Code:
   - Click **"Terminal"** in the top menu
   - Select **"New Terminal"**
2. Type `node --version` and press Enter - you should see a version number
3. Type `npm --version` and press Enter - you should also see a version number

If you see version numbers for both, you're all set!

---

## Part 3: Creating Your React App

### Step 7: Create a React Application

Now we'll create your actual website using React!

1. In the VS Code terminal (at the bottom of the window), make sure you're in your repository folder
2. Type the following command and press Enter:
   ```
   npx create-react-app .
   ```
   - **Note the dot (.) at the end!** This tells it to create the app in the current folder
3. This will take several minutes. You'll see lots of text scrolling by - this is normal!
4. When it's done, you'll see a message like "Happy hacking!"

**What just happened?** Create React App set up everything you need for a React website, including all the code files and tools.

### Step 8: Test Your Website Locally

Let's make sure your website works on your computer before publishing it!

1. In the terminal, type:
   ```
   npm start
   ```
2. Wait a moment - your default web browser should automatically open
3. You should see a spinning React logo and "Edit src/App.js and save to reload"
4. Congratulations! Your React app is running!

To stop the development server:
- Go back to VS Code
- Click in the terminal
- Press `Ctrl + C` (Windows) or `Cmd + C` (Mac)
- Type `Y` if asked to confirm

---

## Part 4: Setting Up GitHub Pages

### Step 9: Install GitHub Pages Package

To publish your site to GitHub Pages, we need one more tool.

1. Make sure your development server is stopped (see end of Step 8)
2. In the terminal, type:
   ```
   npm install --save gh-pages
   ```
3. Press Enter and wait for installation to complete

### Step 10: Configure Your Project for GitHub Pages

We need to tell your project where it will be published.

1. In VS Code's file explorer (left sidebar), click on **`package.json`**
2. At the very top of the file, after the opening `{` and before `"name"`, add this line:
   ```json
   "homepage": "https://yourusername.github.io",
   ```
   - Replace `yourusername` with your actual GitHub username
   - **Important:** Don't forget the comma at the end!
   
3. Scroll down to find the `"scripts"` section (around line 16)
4. Inside the scripts section, add these two lines before the closing `}`:
   ```json
   "predeploy": "npm run build",
   "deploy": "gh-pages -d build"
   ```
   - **Important:** Add a comma to the end of the line above these new lines!

Your scripts section should now look like this:
```json
"scripts": {
  "start": "react-scripts start",
  "build": "react-scripts build",
  "test": "react-scripts test",
  "eject": "react-scripts eject",
  "predeploy": "npm run build",
  "deploy": "gh-pages -d build"
},
```

5. Save the file (`Ctrl + S` on Windows, `Cmd + S` on Mac)

---

## Part 5: Publishing Your Website

### Step 11: Commit Your Changes

"Committing" means saving your changes with a description of what you did.

1. In the terminal, type these commands one at a time, pressing Enter after each:
   ```
   git add .
   ```
   This stages all your changes.
   
   ```
   git commit -m "Initial React app setup"
   ```
   This commits your changes with a message.

**Note:** If this is your first time using Git, you might get an error asking you to set your identity. If so, run these commands first (replace with your information):
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```
Then try the commit command again.

### Step 12: Push Your Changes to GitHub

"Pushing" means uploading your changes to GitHub.

1. In the terminal, type:
   ```
   git push origin main
   ```
   - If you get an error mentioning "master" instead of "main", try:
   ```
   git push origin master
   ```
2. You might be asked to log in to GitHub - enter your username and password
   - **Note:** For the password, you may need to use a personal access token instead of your regular password. If you get an authentication error, follow GitHub's instructions to create a token.

### Step 13: Deploy Your Website

This is the moment of truth! Let's publish your website to the internet.

1. In the terminal, type:
   ```
   npm run deploy
   ```
2. This will take a minute or two. You'll see messages about building and publishing
3. When it's done, you'll see "Published"

### Step 14: Configure GitHub Pages Settings

One last step to make sure everything is connected properly.

1. Go to your repository on GitHub (in your web browser)
2. Click the **"Settings"** tab near the top
3. In the left sidebar, click **"Pages"** (under "Code and automation")
4. Under "Source", you should see "Deploy from a branch"
5. Under "Branch", select **`gh-pages`** from the dropdown (not main!)
6. Make sure **`/ (root)`** is selected next to it
7. Click **"Save"**

---

## Part 6: View Your Live Website!

Your website is now live on the internet!

1. Wait 2-3 minutes for GitHub to process everything
2. Open a web browser and go to: `https://yourusername.github.io`
   - Replace `yourusername` with your actual GitHub username
3. You should see your React website with the spinning logo!

**Congratulations!** You've just published your first website to the internet!

---

## Making Updates to Your Website

Now that your site is live, here's how to make changes and publish them:

### Editing Your Website

1. Open your project in VS Code
2. To edit the main page, open `src/App.js`
3. Make your changes (try changing the text!)
4. Save the file

### Testing Changes Locally

1. In the terminal, run `npm start`
2. Your browser will open showing your changes
3. Press `Ctrl + C` (or `Cmd + C`) to stop the server when done

### Publishing Your Updates

After making changes, follow these steps to publish them:

1. In the terminal, run these commands one at a time:
   ```
   git add .
   git commit -m "Description of what you changed"
   git push origin main
   npm run deploy
   ```
2. Wait 2-3 minutes, then refresh your website to see the changes!

---

## Troubleshooting Common Issues

### "git is not recognized"
- Install Git from [git-scm.com](https://git-scm.com/downloads)
- Restart your terminal

### "npm is not recognized"
- Reinstall Node.js from [nodejs.org](https://nodejs.org)
- Make sure to check "npm package manager" during installation
- Restart your computer

### Website shows a blank page
- Check that your `homepage` in `package.json` has the correct username
- Make sure you selected the `gh-pages` branch in GitHub Pages settings
- Wait a few minutes and try refreshing

### "Permission denied" when pushing to GitHub
- You may need to create a personal access token
- Go to GitHub Settings → Developer settings → Personal access tokens
- Generate a new token and use it as your password

### Changes aren't showing up on the live site
- Make sure you ran `npm run deploy` (not just `git push`)
- Wait 2-3 minutes for GitHub to update
- Try a hard refresh: `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac)

---

## Next Steps

Now that you have a working website, here are some things to try:

1. **Customize your website:** Edit `src/App.js` and `src/App.css` to change the look and content
2. **Learn React:** Check out the [official React tutorial](https://react.dev/learn)
3. **Add more pages:** Learn about React Router to create multiple pages
4. **Make it yours:** Add your projects, resume, or portfolio

Remember: every developer started where you are now. Don't be afraid to experiment and break things - you can always start over!

Happy coding! 🚀
