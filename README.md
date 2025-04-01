# Python Machine Learning Resources

In this guide, I have curated a selection of resources to assist beginners in getting started with Python-based AI projects.

## 1. AI Tools for Productivity (Recommended)
- **[NotebookLM](https://notebooklm.google/):** Leverage AI to summarize sources like web links, research papers, videos, and audio content. It also enables you to ask questions and take notes effectively.
- **[GitHub](https://github.com/):** Manage your code repositories, utilize AI-powered coding assistance, and showcase your coding expertise for career growth.
- **[ChatGPT](https://chatgpt.com/):** Get answers to general queries and enhance your search results with AI-driven insights.

## 2. Setting Up Python Programming Environment
A Python programming environment typically consists of the following components:

- **Linux-like Environment:** A terminal-based setup or tools that emulate a Linux environment for seamless development.
- **Integrated Development Environment (IDE):** Tools like PyCharm, VS Code, or Jupyter Notebook to write, debug, and manage your code efficiently.
- **Packages and Modules:** Libraries and frameworks such as NumPy, Pandas, and Matplotlib to extend Python's functionality for machine learning and data analysis.

### 2.1. Windows - WSL (Only applicable to Windows users)
While Linux and macOS come with a built-in Linux-like environment, Windows users need to install the Windows Subsystem for Linux (WSL) to access a Linux terminal. You can find the installation guide at [Microsoft Learn - Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

Here is a sample procedure to install WSL with an Ubuntu Linux distribution:

1. Open PowerShell or Command Prompt as Administrator: Right-click the Start Menu and select **"Windows PowerShell (Admin)"**.
2. Run the command: `wsl --install`.
3. Once the installation completes, restart your computer.
4. After restarting, search for **"Ubuntu"** in the Windows Start Menu to launch the WSL Ubuntu app.
5. On the first launch, WSL will take some time to initialize. You will then be prompted to set up a username and password for the first user.

### 2.2. Integrated Development Environment (IDE) - Choose one
- **[VS Code](https://code.visualstudio.com/):** Connect to remote SSH servers or WSL environments seamlessly, with support for GitHub's AI-powered coding assistance.
- **[PyCharm Community Edition](https://www.jetbrains.com/pycharm/download):** A Python-focused IDE designed for efficient coding and debugging.

### 2.3. Managing Environments and Packages with Miniconda
Miniconda is a lightweight tool for managing Python environments and packages efficiently.

#### 2.3.1. Installing Miniconda
Follow the quickstart guide to install Miniconda: [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

#### 2.3.2. Package Management Examples
- **Search for a package:**
    ```bash
    conda search scipy
    ```
- **Install packages:**
    ```bash
    conda install numpy scipy pandas matplotlib seaborn scikit-learn
    ```
- **Uninstall a package:**
    ```bash
    conda remove scipy
    ```

#### 2.3.3. Environment Management Examples
Conda environments allow you to maintain isolated setups with specific Python versions and packages. These environments can coexist, and you can easily switch between them. The active environment is displayed as a prefix in your terminal prompt (e.g., `(base)`).

- **Create a new environment:**
    ```bash
    conda create --name myenv python=3.9
    ```
- **Activate an environment:**
    ```bash
    conda activate myenv
    ```
- **Switch back to the previous environment:**
    ```bash
    conda deactivate
    ```
- **List all environments:**
    ```bash
    conda env list
    ```
- **Delete an environment:**
    ```bash
    conda env remove --name myenv
    ```


### 2.4. Version Control with Git (Optional)
Git is a powerful version control system (VCS) that allows developers to track code changes, collaborate effectively, and manage project versions. Most Linux distributions include Git by default. If not, you can install it using:
```bash
sudo aptitude install git
```

#### 2.4.1. Connecting Git to GitHub

1. **Set up your identity:**
    ```bash
    git config --global user.name "<Your Name>"
    git config --global user.email <your email>
    ```

2. **Save credentials temporarily (6 hours):**
    ```bash
    git config --global credential.helper 'cache --timeout=21600'
    ```

3. **Save credentials permanently:**
    ```bash
    git config --global credential.helper store
    ```

#### 2.4.2. Using a GitHub Personal Access Token
When pushing changes to GitHub, you will need a Personal Access Token instead of a password. Tokens are displayed only once during creation. If credentials are stored, the token can be found in the `~/.git-credentials` file. Learn more about creating a token here: [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic).

#### 2.4.3. Helpful Resources
- **[Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf):** A quick reference guide for Git commands.
- **[Git Video Tutorial](https://youtu.be/8JJ101D3knE?si=VtNf4BaDNWxfqfB6):** A beginner-friendly video to learn Git in one hour.
- **[Oh Shit Git](https://ohshitgit.com/):** A resource to help you recover from common Git mistakes.

#### 2.4.4. Commonly Used Git Commands
- Clone a repository: 
  ```bash
  git clone [url]
  ```
- Pull changes from a remote repository: 
  ```bash
  git pull
  ```
- Push changes to a remote repository: 
  ```bash
  git push
  ```
- Push a new branch to a remote repository: 
  ```bash
  git push --set-upstream origin [branchname]
  ```
- Check the status of your repository: 
  ```bash
  git status
  ```
- Add files to staging: 
  ```bash
  git add [file/folder]
  ```
- Switch to an existing branch: 
  ```bash
  git checkout [branchname]
  ```
- Create and switch to a new branch: 
  ```bash
  git checkout -b [branchname]
  ```
- List all branches: 
  ```bash
  git branch -v
  ```
- Delete a branch: 
  ```bash
  git branch -d [branchname]
  ```
- Merge a branch into the current branch: 
  ```bash
  git merge [branchname]
  ```
- Stage and commit changes with a message: 
  ```bash
  git commit -a -m "[commit message]"
  ```

## Introduction to Machine Learning
## Introduction to Machine Learning

Machine learning models learn patterns from input-output data during training and use those patterns to make predictions on new inputs. While some models may seem like a black box, many offer interpretability through techniques like feature importance and decision trees. Neural networks, however, often behave more like a black box compared to traditional machine learning algorithms. 

For an interactive demonstration of how Neural Networks work, check out this [interactive web page](https://playground.tensorflow.org/).

### Video Resources for Basic Machine Learning Concepts
Here is a collection of very short videos that explain basic concepts of Machine Learning. Source: [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)

1. **[A Gentle Introduction to Machine Learning](https://youtu.be/Gv9_4yMHFhI?si=4Vc0WXy5EIzLzmUU)**  
    *Duration: 12 minutes*  
    This video provides a gentle introduction to machine learning, explaining that it is primarily about making predictions and classifications. It emphasizes the crucial role of testing data in evaluating and selecting the best-performing methods, regardless of their complexity.

2. **[Machine Learning Fundamentals: Cross Validation](https://youtu.be/fSytzGwwBVw?si=SbTnOX8W47tvnlnc)**  
    *Duration: 6 minutes*  
    This video explains cross-validation, a method to compare different machine learning techniques and estimate their real-world performance by repeatedly training and testing them on different parts of the data.

3. **[Machine Learning Fundamentals: The Confusion Matrix](https://youtu.be/Kdsp6soqA7o?si=cHPREZ2vvvZWE8pF)**  
    *Duration: 7 minutes*  
    This video introduces the confusion matrix, a fundamental tool in machine learning that summarizes the performance of a classification algorithm by showing the counts of correct and incorrect predictions for each class.

## Introduction to Scikit

## Mini projects

### Regression Example

### Classification Example

### Neural Networks Example

## Plotting and Presenting

### Pandas, Matplotlib and Seaborn

### Jupyter Lab Notebook

