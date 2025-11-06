# 🌟 TrailblazersProjects

Welcome to the **TrailblazersProjects** repository!

The purpose of this repo is to:  
- Host and showcase Trailblazers projects 🧠  
- Provide resources, guides, and instructions for learning Git & GitHub 🧰  
- Serve as a central place for students to submit assignments and code 💻

---

## 🚀 Getting Started

If you're new to Git and GitHub follow the steps below to get everything set up.  

### 1. Create a GitHub Account
1. Go to [https://github.com/](https://github.com/).  
2. Click **Sign up** and create an account using your school or personal email.  
3. Verify your email and set your username (make it professional — this is visible to others).

---

## 📤 Submitting Work (For All Teams)

**Mechanical & Electrical Teams:** Use this simple browser-based method!  
**Software & Data Teams:** You can use this method OR the command line method below.

### Step 1: Fork the Repository

1. Go to the [TrailblazersProjects repository](https://github.com/YOUR_ORG/TrailblazersProjects)
2. Click the **Fork** button in the top-right corner
3. This creates your own copy of the repository under your GitHub account

### Step 2: Upload Your Files

1. Go to **your forked repository** (it should be at `github.com/YOUR_USERNAME/TrailblazersProjects`)
2. Navigate to the folder where you want to add files (e.g., `assignments/`)
3. Click **Add file** → **Upload files**
4. Drag and drop your files or click to browse
5. Scroll down and add a commit message like "Add assignment 1 submission"
6. Click **Commit changes**

### Step 3: Create a Pull Request

1. Go back to your forked repository main page
2. Click the **Contribute** button (or **Pull requests** tab)
3. Click **Open pull request**
4. Add a title like "Assignment 1 - Your Name"
5. Add a description explaining what you're submitting
6. Click **Create pull request**
7. Done! An instructor will review and merge your work

---

## 💻 Command Line Git (Software & Data Teams)

If you prefer using the command line for more advanced workflows, follow these steps:

### Installing Git

#### 🪟 Windows:
- Download and install Git from [https://git-scm.com/download/win](https://git-scm.com/download/win).  
- During setup, you can keep most default settings.  
- After installation, open **Git Bash** to verify it worked.

#### 🍎 macOS:
- Install Git using Homebrew:
  ```bash
  brew install git
  ```
- Or install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

#### 🐧 Linux:
- Install Git using your package manager:
  ```bash
  sudo apt-get install git  # Ubuntu/Debian
  ```

#### ✅ Verify Installation:
Open your terminal (or Git Bash on Windows) and run:
```bash
git --version
```

### Configure Git

Set up your identity:
```bash
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
```

### Basic Git Workflow

**Clone the repository:**
```bash
git clone https://github.com/YOUR_ORG/TrailblazersProjects.git
cd TrailblazersProjects
```

**Create a new branch:**
```bash
git checkout -b your-name-assignment1
```

**Make your changes, then stage and commit:**
```bash
git add .
git commit -m "Add assignment 1 solution"
```

**Push to GitHub:**
```bash
git push origin your-name-assignment1
```

**Then create a pull request on GitHub as described above!**

---

## 📚 Additional Resources

- [GitHub's Hello World Tutorial](https://guides.github.com/activities/hello-world/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)

---

## 🆘 Need Help?

If you run into any issues or have questions:
- Ask in the Trailblazers GroupMe or NSBE Discord
- Reach out to your team lead

---

## 🎯 Project Structure

```
TrailblazersProjects/
├── README.md
├── Tutorials/
│   ├── tutorial-1/
│   ├── tutorial-2/
│   └── ...
└── Assignments/
    ├── assignment-1/
    ├── assignment-2/
    └── ...
```

---
