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
<details>
<summary>Click here to expand</summary>
While Linux and macOS come with a built-in Linux-like environment, Windows users need to install the Windows Subsystem for Linux (WSL) to access a Linux terminal. You can find the installation guide at [Microsoft Learn - Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

Here is a sample procedure to install WSL with an Ubuntu Linux distribution:

1. Open PowerShell or Command Prompt as Administrator: Right-click the Start Menu and select **"Windows PowerShell (Admin)"**.
2. Run the command: `wsl --install`.
3. Once the installation completes, restart your computer.
4. After restarting, search for **"Ubuntu"** in the Windows Start Menu to launch the WSL Ubuntu app.
5. On the first launch, WSL will take some time to initialize. You will then be prompted to set up a username and password for the first user.
</details>

### 2.2. Integrated Development Environment (IDE) - Choose one
<details>
<summary>Click here to expand</summary>
- **[VS Code](https://code.visualstudio.com/):** Connect to remote SSH servers or WSL environments seamlessly, with support for GitHub's AI-powered coding assistance.
- **[PyCharm Community Edition](https://www.jetbrains.com/pycharm/download):** A Python-focused IDE designed for efficient coding and debugging.
</details>

### 2.3. Managing Environments and Packages with Miniconda
<details>
<summary>Click here to expand</summary>
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
</details>

### 2.4. Version Control with Git (Optional)
<details>
<summary>Click here to expand</summary>
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
</details>

## 3. Introduction to Machine Learning

Machine learning models learn patterns from input-output data during training and use those patterns to make predictions on new inputs. While some models may seem like a black box, many offer interpretability through techniques like feature importance and decision trees. Neural networks, however, often behave more like a black box compared to traditional machine learning algorithms. 

For an interactive demonstration of how Neural Networks work, check out this [interactive web page](https://playground.tensorflow.org/).

### 3.1. Video Resources for Basic Machine Learning Concepts
<details>
<summary>Click here to expand</summary>
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
4. **[Machine Learning Fundamentals: Sensitivity and Specificity](https://youtu.be/vP06aMoz4v8?si=FIgghaSUcheQZySW)**  
    *Duration: 11 minutes*    
    This video demonstrates how to compute and understand sensitivity, the proportion of true positive cases correctly identified, and specificity, the proportion of true negative cases correctly identified. It uses confusion matrices with multiple categories to assess the performance of machine learning models.
5. **[The Sensitivity, Specificity, Precision, Recall Sing-a-Long!!!](https://youtu.be/PWvfrTgaPBI?si=Da5HjoP68kXv1D1t)**  
    *Duration: 1 minute*  
    This video revisits the following key metrics in machine learning:

    - **Sensitivity:** The proportion of actual positives that are correctly identified.
    - **Specificity:** The proportion of actual negatives that are correctly identified.
    - **Precision:** The proportion of predicted positives that are correctly identified.
6. **[Machine Learning Fundamentals: Bias and Variance](https://youtu.be/EuBBz3bI-aA?si=TgLSCDKtwAndVyQl)**  
    *Duration: 6 minutes*  
    This video explains the fundamental machine learning concepts of bias, the inability of a model to capture the true relationship in data, and variance, the sensitivity of a model's performance to different datasets, using the example of fitting lines to predict mouse height from weight.
7. **[ROC and AUC, Clearly Explained!](https://youtu.be/4jRBRDbJemM?si=6ssJLNSbJqmXvkq3)**  
    *Duration: 16 minutes*  
    This video explains that ROC graphs visualize a classification model's performance across different classification thresholds by plotting the true positive rate against the false positive rate, while the AUC provides a single numerical value representing the overall performance of the model.
8. **[Entropy (for data science) Clearly Explained!!!](https://youtu.be/YtebGVx-Fxw?si=I4ffKZnbdgk9Hjk_)**  
    *Duration: 16 minutes*  
    This video explains entropy in data science as the expected value of surprise. It quantifies similarities and differences by defining surprise as the logarithm of the inverse of probability.
9. **[Mutual Information, Clearly Explained!!!](https://youtu.be/eJIp_mgVLwE?si=XxufG7nZUTpjjpKT)**  
    *Duration: 16 minutes*  
    This video describes mutual information as a numerical measure that evaluates the relationship between two variables by analyzing their joint and individual probabilities. Mutual Information measures any dependency (including non-linear relationships), while Correlation only measures linear or monotonic relationships.
10. **[The Main Ideas of Fitting a Line to Data (The Main Ideas of Least Squares and Linear Regression.)](https://youtu.be/PaFPbb66DxQ?si=5DPSiUmcS8PjkI_N)**  
    *Duration: 9 minutes*  
    This video explains linear regression, also referred to as least squares, as a method to find the best-fitting line for a dataset by minimizing the total of the squared vertical distances between the data points and the line.
11. **[Linear Regression, Clearly Explained!!!](https://youtu.be/nk2CQITm_eo?si=2f7Gaf2Vano72dz0)**  
    *Duration: 27 minutes*  
    This video explains how linear regression applies the least squares method to fit a line (or a plane in higher dimensions) to the data. It also discusses how to measure the fit's strength using R-squared and evaluates the statistical significance of R-squared through a p-value derived from the F-statistic.
12. **[Multiple Regression, Clearly Explained!!!](https://youtu.be/zITIFTsivN8?si=qTrRwv0kBz0Vr_oq)**  
    *Duration: 5 minutes*  
    This video explains multiple regression as an extension of simple linear regression. It models a dependent variable using multiple independent variables by fitting a plane or a higher-dimensional surface to the data. The method also evaluates the impact of additional variables by comparing models with and without them using R-squared values and p-values.
13. **[Using Linear Models for t-tests and ANOVA, Clearly Explained!!!](https://youtu.be/NF5_btOaCig?si=NXSKr5hX9_u4Tliz)**  
    *Duration: 11 minutes*  
    This video explains how the principles of linear regression, especially the use of a design matrix, can be extended to conduct t-tests and ANOVA. It demonstrates how to fit means to various groups and compute p-values using an F-statistic based on the sum of squared residuals.
14. **[Odds and Log(Odds), Clearly Explained!!!](https://youtu.be/ARfXDSkQf1Y?si=GcahliJdczX7cu-d)**  
    *Duration: 11 minutes*  
    Odds represent the ratio of the likelihood of an event happening to the likelihood of it not happening. The logarithm of the odds transforms this ratio into a scale centered around zero, making it more interpretable and useful in statistical models like logistic regression.
15. **[Odds Ratios and Log(Odds Ratios), Clearly Explained!!!](https://youtu.be/8nm0G-1uJzA?si=h6S62AHf1oNQfc7X)**  
    *Duration: 16 minutes*  
    This video explains odds ratios as a measure of association between two events, calculated as the ratio of their odds. The logarithm of the odds ratio offers a symmetric scale, and its statistical significance can be assessed using tests such as Fisher's exact test, chi-square test, or Wald test.
16. **[StatQuest: Logistic Regression](https://youtu.be/yIYKR4sgzI8?si=fxMSQVADi3XHbPAr)**  
    *Duration: 8 minutes*  
    This video introduces logistic regression, a machine learning method akin to linear regression, designed to estimate the probability of a binary outcome (e.g., true or false). It employs an s-shaped logistic function and can handle both continuous and categorical data for classification tasks.
17. **[Logistic Regression Details Pt1: Coefficients](https://youtu.be/vN5cNN2-HWE?si=fo9ZSAEeRSkAF6e_)**  
    *Duration: 19 minutes*  
    This video explains logistic regression coefficients, which predict the log odds of a binary outcome. These coefficients indicate the change in log odds for a one-unit increase in a continuous predictor or the log odds ratio for a categorical predictor. Their statistical significance is evaluated using standard errors and Z-values (Wald's test).
18. **[Logistic Regression Details Pt 2: Maximum Likelihood](https://youtu.be/BfKanl1aSG0?si=XaTLACfNe_Basj9o)**  
    *Duration: 10 minutes*  
    This video explains how logistic regression fits curves to data by using a method called maximum likelihood, which identifies the curve that best predicts the observed binary outcomes. Unlike least squares, this approach is necessary due to the transformed y-axis and the presence of infinite residuals.
19. **[Logistic Regression Details Pt 3: R-squared and p-value](https://youtu.be/xxFYro8QuXA?si=wy81TIDYrqVMZK-8)**  
    *Duration: 15 minutes*  
    This video discusses the use of R-squared and p-values to evaluate the fit and significance of logistic regression models. It highlights the complexity of calculating R-squared for logistic regression, as there is no universally accepted method. The video focuses on McFadden's pseudo R-squared, which is derived from the log-likelihoods of the fitted model and the intercept-only model. Additionally, it explains the use of a chi-squared test based on the difference in log-likelihoods to compute the p-value.
20. **[Saturated Models and Deviance](https://youtu.be/9T0wlKdew6I?si=S8wq80dgSUugP2Lb)**
    *Duration: 18 minutes*
    This video clarifies the concept of the saturated model—one that includes a parameter for every data point—and explains how it sets an upper limit for the log-likelihood-based R-squared. It also explores the model's relationship to deviance statistics (residual and null deviance) used in hypothesis testing, while pointing out that it has no practical relevance in logistic regression.
21. **[Deviance Residuals](https://youtu.be/JC56jS2gVUE?si=c5P7922fCyrmhTtP)**  
    *Duration: 6 minutes*  
    Deviance residuals indicate the square root of each data point's contribution to the total residual deviance and are used to detect outliers.
22. **[Regularization Part 1: Ridge (L2) Regression](https://youtu.be/Q81RR3yKn30?si=bNu8I3XkUC2Gbanx)**  
    *Duration: 20 minutes*  
    This video explains Ridge regression, a regularization method that modifies traditional least squares by adding a penalty term. This approach reduces variance and enhances prediction accuracy, especially in cases with small datasets or limited features, by introducing slight bias and shrinking parameter estimates.

</details>

### 3.2. Book Resources for Machine Learning Geeks and Historians (Optional)
<details>
<summary>Click here to expand</summary>

**[PATTERNS, PREDICTIONS, AND ACTIONS: A Story About Machine Learning](https://arxiv.org/pdf/2102.05242)**  
This book provides a thorough introduction to machine learning, starting with the basics of prediction and progressing to advanced topics such as deep learning and causal inference. It highlights the importance of datasets and benchmarks in the field, offering a modern perspective on causality and sequential decision-making—areas often overlooked in traditional machine learning courses. Additionally, the authors explore the potential harms and societal implications of machine learning technologies, fostering a deeper understanding of its impact beyond pattern recognition.

This book requires certain Mathematics background to read.
</details>

## 4. Introduction to Scikit-learn
Scikit-learn is a widely-used Python library for machine learning, offering efficient and user-friendly tools for data analysis and modeling. Built on top of NumPy, SciPy, and Matplotlib, it supports a variety of algorithms for tasks such as classification, regression, clustering, and dimensionality reduction.

With its intuitive API, scikit-learn makes it easy to build and evaluate machine learning models in just a few lines of code. For instance, training a decision tree classifier can be done as follows:

```python
from sklearn.tree import DecisionTreeClassifier  
model = DecisionTreeClassifier()  
model.fit(X_train, y_train)  
predictions = model.predict(X_test)
```

Scikit-learn is highly regarded in both academic and professional settings, making it an excellent choice for anyone looking to dive into machine learning. If you have a basic understanding of Python and want to tackle real-world data challenges, scikit-learn is a fantastic starting point!

### 4.1. Mini projects

### 4.2. Regression Example

### 4.3. Classification Example

### 4.4. Neural Networks Example

## Plotting and Presenting

### Pandas, Matplotlib and Seaborn

### Jupyter Lab Notebook

