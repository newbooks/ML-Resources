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

The focuses here are:
- Basic concepts
- Regression
- Principal Component Analysis
- Clustering
- Classification

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
16. **[Logistic Regression](https://youtu.be/yIYKR4sgzI8?si=fxMSQVADi3XHbPAr)**  
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
22. **[Principal Component Analysis (PCA), Step-by-Step](https://youtu.be/FgakZw6K1QQ?si=EqIIaIyjPv0REEft)**  
    *Duration: 21 minutes*  
    This video provides a detailed explanation of Principal Component Analysis (PCA), a dimensionality reduction technique that identifies principal components—linear combinations of variables—that capture the maximum variance in the data. It also covers key concepts such as eigenvalues and loading scores, enabling visualization and identification of significant variables.
23. **[StatQuest: PCA main ideas in only 5 minutes!!!](https://youtu.be/HMOI_lkzW08?si=f1ShHAppnstUybe8)**  
    *Duration: 6 minutes*  
    This video provides a concise explanation of Principal Component Analysis (PCA), a technique for reducing the dimensionality of multi-variable datasets. It demonstrates how PCA transforms data into a 2D plot, with axes representing the most significant principal components, to reveal patterns and clusters.
24. **[PCA - Practical Tips](https://youtu.be/oRvgq966yZg?si=jN6JfKWCQTlw4Vlz)**  
    *Duration: 8 minutes*  
    This video offers practical advice for conducting Principal Component Analysis (PCA), highlighting the importance of scaling and centering data for reliable outcomes. It also explains how the number of samples and variables influences the maximum possible principal components.
25. **[PCA in Python](https://youtu.be/Lsue2gEM9D0?si=xlMgoos-7vjWoipy)**  
    *Duration: 11 minutes*  
    This video provides a step-by-step guide to performing Principal Component Analysis (PCA) in Python. It demonstrates key processes such as data preparation (scaling and centering), computing principal components, visualizing results using scree and scatter plots, and analyzing loading scores with libraries like scikit-learn and matplotlib.
26. **[Hierarchical Clustering](https://youtu.be/7xHsRkOdVwo?si=WzWyVVTpPV-7DMm_)**  
    *Duration: 11 minutes*  
    This video provides an explanation of hierarchical clustering, a technique commonly used alongside heat maps to organize rows and columns based on similarity. It describes the iterative process of merging the most similar genes or clusters and explores various methods for measuring similarity, such as Euclidean and Manhattan distances. Additionally, it discusses approaches for comparing clusters, including centroid, single linkage, and complete linkage, with the resulting hierarchy visualized through a dendrogram.
27. **[K-means clustering](https://youtu.be/4b5d3muPQmA?si=-_inBEYT5NeUyuSv)**  
    *Duration: 8 minutes*  
    This video explains K-means clustering, an iterative algorithm that groups data points by assigning them to the nearest cluster center and updating the centers until convergence. It demonstrates the method using examples like line plots, XY graphs, and heat maps, and discusses how to determine the optimal number of clusters (K) using an elbow plot.
28. **[K-nearest neighbors, Clearly Explained](https://youtu.be/HVXime0nQeI?si=UgO92xIZjVtv1O-Y)**  
    *Duration: 5 minutes*  
    This video provides an overview of the K-nearest neighbors algorithm, a classification technique that predicts the category of a new data point by identifying its K closest neighbors in the training dataset and assigning it the most common category among them.
29. **[Naive Bayes](https://youtu.be/O2L2Uv9pdDA?si=seI16Wkg4nN8J0gW)**  
    *Duration: 15 minutes*  
    This video provides an overview of the multinomial Naive Bayes classifier, showcasing its application in spam message filtering. It explains how the algorithm calculates the likelihood of words appearing in normal and spam messages, combines these with prior probabilities to compute a score for each category, and classifies a new message based on the higher score. The video also highlights the "naive" assumption of treating word order as irrelevant.
30. **[Gaussian Naive Bayes](https://youtu.be/H3EjCKtlVog?si=c4PBtcgIAkhZ8kgK)**  
    *Duration: 9 minutes*  
    This video provides an overview of Gaussian Naive Bayes, a classification algorithm that models features using Gaussian distributions. It calculates category scores by combining prior probabilities with feature likelihoods, often using logarithms to avoid underflow issues. The method is demonstrated through an example of predicting movie preferences.
31. **[Decision and Classification Trees](https://youtu.be/_L39rN6gz7Y?si=UXKiUgtWLLBfzhu0)**  
    *Duration 18 minutes*  
    This video provides an overview of classification trees, explaining their construction through iterative splits of data based on features that reduce impurity (commonly measured using Gini impurity) until reaching leaf nodes that determine the final category. It also demonstrates how these trees classify new data.  
32. **[Decision Trees, Part 2 - Feature Selection and Missing Data](https://youtu.be/wpNl-JwwplA?si=W6EE2v8VTnuYaxdx)**  
    *Duration: 5 minutes*  
    This video delves into feature selection in decision trees, highlighting how features are chosen based on their ability to reduce impurity. It also discusses setting thresholds to prevent overfitting and explores strategies for handling missing data, such as imputing with the most frequent value, a correlated feature, the mean or median, or using linear regression to estimate missing values.
33. **[Regression Trees](https://youtu.be/g9c66TUylZ4?si=hZFY8aLORCHxZ4cQ)**  
    *Duration: 22 minutes*  
    This video explains regression trees, a type of decision tree designed for numeric predictions. It details the process of building these trees by iteratively splitting data based on thresholds that minimize the sum of squared residuals. Each leaf node represents the average numeric value of the observations in that group. To prevent overfitting, splitting stops when a node contains fewer than a specified minimum number of observations.
34. **[How to Prune Regression Trees](https://youtu.be/D0efHEJsfHo?si=187pAeYwaoKvbLh0)**  
    *Duration: 16 minutes*  
    This video explains cost complexity pruning (also known as weakest link pruning) for regression trees. It demonstrates how to prevent overfitting by calculating a tree's score using the sum of squared residuals combined with a complexity penalty based on the number of leaves. The video also covers using cross-validation to determine the optimal tuning parameter (alpha) that minimizes the sum of squared residuals on test data.
35. **[Classification Trees in Python from Start to Finish](https://youtu.be/q90UDEgYqeI?si=NSERYgCcdCoFQzAz)**  
    *Duration: 66 minutes*  
    This video provides a comprehensive walkthrough of building and optimizing a classification tree in Python using scikit-learn. It covers key steps such as importing and preparing data, addressing missing values and categorical data through one-hot encoding, splitting the dataset, constructing an initial tree, and applying cost complexity pruning with cross-validation to develop a refined tree for predicting heart disease.
36. **[Random Forests Part 1 - Building, Using and Evaluating](https://youtu.be/J4Wdy0Wc_xQ?si=Y65Sl7t6PrQKf9Cx)**  
    *Duration: 10 minutes*  
    This video covers the fundamentals of random forests, demonstrating how they enhance the accuracy of decision trees by aggregating predictions from multiple trees trained on bootstrapped samples and random feature subsets.
37. **[Random Forests Part 2: Missing data and clustering](https://youtu.be/sQ870aTKqiM?si=vd5cmrtsLrbQKN48)**  
    *Duration: 12 minutes*  
    This video describes how random forests address missing data by iteratively updating initial estimates using sample similarities measured with a proximity matrix. It also explains how the proximity matrix can be leveraged for clustering and visualizing samples.
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

**Install Machine Learning and other modules used by this tutorial:**  
Assuming we manage our Python, packages and programming environment with [miniconda](#23-managing-environments-and-packages-with-miniconda), we can install scikit-learn and its dependency packages with commands:

```bash
conda install scikit-learn seaborn jupyterlab
```


### 4.1. Mini projects
Below are common applications of machine learning.

### 4.2. Regression Example
<details>
<summary>Click here to expand</summary>

In this project, we will
1. View the input data  
   - Display summary statistics.
   - Visualize key features (e.g., histograms, scatterplots, heatmaps).

2. Data Preprocessing  
   - Normalize/scale features if necessary.
   - Split data into training and test sets.

3. Model Training. Find at least two regression models to train, such as:
   - Linear Regression
   - Ridge or Lasso Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Support Vector Regression (SVR)

4. Model Evaluation
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R<sup>2</sup>

5. Visualize predictions vs. actual values

**Example procedure:**

- A step by step interactive implementation is in a Jupyter Notebook regression/regression.ipynb 
- A working python scrip example is in regression/regression.py


They are self-explanatory.

**A real world example**
A real world example is in folder regression. The code ele_models.py evaluates several regressors in fitting electric potential by contributing factors.

To run the code:
```
./ele_models.py state*.csv
```

</details>


### 4.3. PCA Analysis

### 4.4. Clustering

### 4.5. Classification Example

### 4.6. Neural Networks Example

## Plotting and Presenting

### Pandas, Matplotlib and Seaborn

### Jupyter Lab Notebook

