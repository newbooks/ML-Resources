# Python Machine Learning Resources

In this guide, I have curated a selection of resources to assist beginners in getting started with Python-based AI projects.

## AI Tools for Productivity (Recommended)
- **[NotebookLM](https://notebooklm.google/):** Leverage AI to summarize sources like web links, research papers, videos, and audio content. It also enables you to ask questions and take notes effectively.
- **[GitHub](https://github.com/):** Manage your code repositories, utilize AI-powered coding assistance, and showcase your coding expertise for career growth.
- **[ChatGPT](https://chatgpt.com/):** Get answers to general queries and enhance your search results with AI-driven insights.

## Setting Up Python Programming Environment
A Python programming environment typically consists of the following components:

- **Linux-like Environment:** A terminal-based setup or tools that emulate a Linux environment for seamless development.
- **Integrated Development Environment (IDE):** Tools like PyCharm, VS Code, or Jupyter Notebook to write, debug, and manage your code efficiently.
- **Packages and Modules:** Libraries and frameworks such as NumPy, Pandas, and Matplotlib to extend Python's functionality for machine learning and data analysis.

### Windows - WSL (Only applicable to Windows users)
While Linux and macOS come with a built-in Linux-like environment, Windows users need to install the Windows Subsystem for Linux (WSL) to access a Linux terminal. You can find the installation guide at [Microsoft Learn - Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

Here is a sample procedure to install WSL with an Ubuntu Linux distribution:

1. Open PowerShell or Command Prompt as Administrator: Right-click the Start Menu and select **"Windows PowerShell (Admin)"**.
2. Run the command: `wsl --install`.
3. Once the installation completes, restart your computer.
4. After restarting, search for **"Ubuntu"** in the Windows Start Menu to launch the WSL Ubuntu app.
5. On the first launch, WSL will take some time to initialize. You will then be prompted to set up a username and password for the first user.

### Integrated Development Environment (IDE) - Choose one
- **[VS Code](https://code.visualstudio.com/):** Connect to remote SSH servers or WSL environments seamlessly, with support for GitHub's AI-powered coding assistance.
- **[PyCharm Community Edition](https://www.jetbrains.com/pycharm/download):** A Python-focused IDE designed for efficient coding and debugging.

### Managing Environments and Packages with Miniconda
Miniconda is a lightweight tool for managing Python environments and packages efficiently.

#### Installing Miniconda
Follow the quickstart guide to install Miniconda: [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

#### Package Management Examples
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

#### Environment Management Examples
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


### Manage Version Control (Optional)
Git is a version control system (VCS) that helps developers track changes in their code, collaborate with others, and manage different versions of a project efficiently. Most Linux distributions come with git. If not, install git with:
```
sudo aptitude install git
```

Connect local git to GitHub:

Set up identity:
`$ git config --global user.name "<Your Name>"`
`$ git config --global user.email <your email>`

Save credential for 6 hours:
`$ git config --global credential.helper 'cache --timeout=21600'`

Save credential permanently:
`$ git config --global credential.helper store`


[GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic):

GitHub token is only presented one time at the creation. If a workstation is set top store the credential, the token can be found in `~/.git-credentials` file.

Git Cheatsheet:

Git Video Tutorial:

Git moment - now you know why you need git:
[Oh Shit Git](https://ohshitgit.com/)



## Introduction to Machine Learning

## Introduction to Scikit

## Mini projects

### Regression Example

### Classification Example

### Neural Networks Example

## Plotting and Presenting

### Pandas, Matplotlib and Seaborn

### Jupyter Lab Notebook

