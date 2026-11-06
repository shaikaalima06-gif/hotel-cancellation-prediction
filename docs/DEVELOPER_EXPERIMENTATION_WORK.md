\# Developer 4 - Experimentation Workflow



\## Responsibility

Develop and maintain the machine learning experimentation and model comparison workflow.



\## Experimentation Activities

\- Compare Logistic Regression, Decision Tree, and Random Forest.

\- Track model parameters and configurations.

\- Record accuracy, precision, recall, F1-score, and ROC-AUC.

\- Track experiments using MLflow.

\- Compare model performance.

\- Identify the best-performing model.



\## Experimentation Process

1\. Load the preprocessed training and testing datasets.

2\. Configure the candidate machine learning models.

3\. Train each model using the same dataset split.

4\. Evaluate each model using consistent metrics.

5\. Log parameters, metrics, and artifacts in MLflow.

6\. Compare the experimental results.

7\. Select the best-performing model.



\## Experiment Results

The previous model comparison recorded the following F1-scores:



\- Logistic Regression: 0.7304

\- Decision Tree: 0.7954

\- Random Forest: 0.8574



Random Forest achieved the highest F1-score in that comparison.



\## Result

The experimentation workflow is maintained independently in the experimentation branch.

