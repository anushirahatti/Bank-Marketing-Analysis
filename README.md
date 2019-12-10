# SUMMARY
This Bank Marketing Analysis project is very unique in itself even though this dataset is widely used for implementing Machine Learning projects. In addition to implementation of standard tasks in Data Mining project, the goal of this project is to incorporate **Bias Detection and Mitigation strategy, Data Pipelining, Ensemble methods, Feature Selection techniques, and Model Evaluation methods** in the process of training and implementing a robust and unbiased Machine Learning model. We have used cross-industry standard process for data mining.

We did not require much efforts to clean the data because the data quality was reasonalbly good. Since the dataset was unbalanced, we used oversampling technique to balance the dataset before data preprocessing. As we are using multiple machine learning models in this project, we have pipelined the task of scaling and modelling to determine which model performs better. We achieved it by implementing two-step Data pipeline:
- **Step 1:** Uses Transformer
- **Step 2:** Uses Estimator

To avoid overfitting the data to training dataset, we have used cross-validation, feature selection, and ensemble methods (Bagging and Boosting). Among all the models we implemented (logistic regression, decision tree classifier, random forest classifier, and support vector machines), Random Forest Classifier performed well. For gaining confidence in the implemented Machine Learning model, we have used evaluation techniques like Confusion Matrix, F1-score, and AUC - ROC.

### Bias Detection and Mitigation strategy:
We have one protected feature in our dataset i.e. "AGE". This raises two problems: 
- the training dataset may not be representative of the true population of people of all age groups, and 
- even if it is representative, it is illegal to base any decision on a applicant's age, regardless of whether this is a good prediction based on historical data. 

To check the fairness, **Disparate Impact metric** is calculated for the protected attribute "AGE". We observed that the dataset has bias in "AGE". The privileged group (older than 25 years) have approximately 200% more positive outcomes in training dataset.
Since, Bias Detection and Mitigation is still area of research, we have to come up with the strategies that will help us train model with the data that is fair and makes unbiased predictions.

### Future Work:
- We plan to extend this project by continuing research on Bias detection and Mitigation.
- We are willing to incorporate pipelining on a large scale and automate most of the standard processes in data mining.
- We are curious to obtain more data from multiple sources and implement Data pipelining with the help of workflow management tools like Luigi and Airflow.

### Instructions for individuals that may want to use our work:
To know more about the work we are doing and for knowing about our contact details, kindly refer **"ProjectINFO.md"** file in the Project repository. We are willing to collaborate and exchange ideas. Thank you!

