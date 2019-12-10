# Bank-Marketing-Analysis
Knowledge Discovery in Databases (Course Project - Team 8)


### Project Name/Intro:
##### - Project Name: Kdd_project_8 (Bank Marketing Analysis)
##### - Project Introduction:
Most marketing is a means of introducing and enforcing a value to purchasers - consumer or business. As to bank marketing: marketing managers who work in corporate banking are responsible for co-ordinating and enforcing branding guidelines and standards across departments and business lines. These professionals focus on generating ideas and content for sales collateral, advertising, external websites and printed signage for events. Other duties include conducting competitive analysis, organizing direct mail campaigns, sending business and news updates to banking clients and managing online promotions. This project is about direct marketing campaign (phone calls) to convince clients to subscribe for a term deposit. Our goal is to classify whether the client will subscribe for the term deposit or not by discovering knowledge in the Bank marketing dataset and making predictions.

### Dataset Link: 
[Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

### Team Introduction:
Aniruddha Sudhindra Shirahatti, Chandrakanth Rajesh, Krishna Vishwanatham, Digvijay Gole, Yash Bonde.
We are having common interest in pursuing specialization in the field of Data Science. We as a group are planning to work on Bank Marketing Data Set to implement the data mining concepts.

### Data and Source description:-
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

Input variables:
-bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
- related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
- other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
- social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

- Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')


### Core Technical Concepts

#### - Application of the CRISP-DM Process:
**1. Business/Research Understanding Phase -** In this phase, we have explored various resources on the internet to learn about the challenges and applications of the Bank marketing dataset for data mining. We did research on how the Bank marketing dataset can be used to uncover knowledge in various domains like branding and marketing,lending loans or capitals,mortgage values,and management, as well as public and personalized insurances .

**2. Data Understanding Phase -** In this phase, we will make observations by looking at the data and checking the relationship among the variables, potential independent and dependant variables, continuous variables, flag variables, categorical variables, mean, median and standard deviation in the data features.

**3. Data Preparation Phase -** In this phase, we will deal with obsolete/redundant fields, missing values, outliers and make the data usable to feed as an input to the data mining models. Thus minimizing GIGO (Garbage In - Garbage Out)

**4. Modeling Phase -** We will fit different data mining models on the preprocessed dataset, check the results and try fine tuning the models to achieve better performance.

**5. Evaluation Phase -** We will try to fit the dataset on various data mining models and evaluate their performance by implementing concepts like Cross-validation, Confusion matrix, etc. If the model performance is good and the results are as per the business understanding requirement, we will deploy, else we will repeat the step 1 to 5 repeatedly till we achieve the desired results.

**6. Deployment Phase -** In this phase, we will deploy the model so that the intended users can incorporate our product for their business requirements.

### Domain Knowledge (document sources):
[DOMAIN-SPECIFIC KNOWLEDGE FOR BANK MARKETING](https://www.omicsonline.org/open-access/marketing-of-financial-and-banking-products-an-example-frombangladeshi-bank-2168-9601-1000159.php?aid=76106)



### Data Understanding :
Marketing of bank products is the aggregate function absorbed at providing facility to satisfy customer’s monetary needs and wants, more than the rivalry keeping in view the organizational objectives. Banking is a personalized service oriented industry and hence should provide services which satisfy the customers needs. The marketing tactic includes forestalling, classifying, responding and satisfying the customers’ needs and wants effectually, professionally, and beneficially. It can be said that the presence of the bank has miniature value without the presence of the customer. The main role of the bank is not only to attain and win more and more customers but also to preserve them through operative customer facility. Marketing as associated to banking is to explain a suitable promise to a customer through a variety of products and services and also to confirm operative distribution through satisfaction. The actual contentment delivered to a customer relay on how the customer is cooperated with. It goes on to prominence that every employee from the highest executive to the most junior employee of the bank should be concerned with marketing.

The dataset is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.


#### - Exploring protected features for Bias (Bias Detection and Mitigation)
A machine learning model makes predictions of an outcome for a particular instance. The model makes these predictions based on a training dataset, where many other instances and actual outcomes are provided. Thus, a machine learning algorithm will attempt to find patterns, or generalizations, in the training dataset to use when a prediction for a new instance is needed. In many domains this technique, called supervised machine learning, has worked very well.

However, sometimes the patterns that are found may not be desirable or may even be illegal. These are due to some features which are termed as protected variables (which have high chances of introducing bias in predictions). We have one such feature in our dataset i.e. "AGE". This raises two problems: 
1. the training dataset may not be representative of the true population of people of all age groups, and 
2. even if it is representative, it is illegal to base any decision on a applicant's age, regardless of whether this is a good prediction based on historical data.

To check the fairness metrics, we have used AI Fairness 360 which is designed to help address this problem. Fairness metrics can be used to check for bias in machine learning workflows.
 



### Data Preparation:
Steps To Prepare The Data.
1. Get the dataset and import the libraries.
2. Handle missing data.
3. Encode categorical data.
4. Splitting the dataset into the Training set and Test set.
5. Feature Scaling, if all the columns are not scaled correctly.

So, we will implement all the steps on the dataset one by one and prepare the final dataset on which we can apply classification algorithms.

1. Get The Dataset.
We need to extract data from anyone API’s. But the data could be in any form. So we need to convert it to the CSV format. CSV stands for Comma Separated Values.
These are some of the libraries that we are going to use for this project:
Numpy
Matplotlib
Pandas
Sklearn

2. Handle Missing Values
When we convert the dataset to the CSV format and get the information about the data, it will have missing values which is usually represented by NA. There are many ways to handle missing values. Whenever we come across minute missing values we are going to drop the rows using the .dropna function. Whenever we come across large missing values we are going to perform KNN Imputation and different interpolation methods to handle the missing values.

3. Encode categorical data.
As we come across categorical data, we need to encode them into numerical format to proceed further to make the analysis. For this we need to encode the data. We are going to use the one-hot encoding to encode the categorical variables when we encounter more than 2 variables. Whenever we come across one or two variables we are going to use the label encoding. We import LabelEncoder and OneHotEncoder from sklearn.preprosseing.

4. Split the Dataset into Training and Test Set.
We have to feed our Data Model Training and test datasets. Generally, we split the data with a ratio of 70% for the Training Data and 30% to test data. Training Data is used to build the m model where as Test set is used to evaluate the model.  From sklearn.model_selection we are going to import train_test_split.

5. Feature Scaling
In a general scenario, Machine Learning is based on Euclidean distance. In our dataset we will be encountering different coulmns with different range of values. That is why this is called feature scaling. We use StandardScaler from sklearn.preprocessing to perform the scaling.

### Machine Learning:
The algorithm that we have used:

* Random Forest : Random Forest is a supervised classification algorithm. This algorithm helps to create a Forest and make it random. There is a direct relationship between the number of trees in the forest and the results it can get: the larger the number of trees, the more accurate the result. But one thing to note is that creating the forest is not the same as constructing the decision with information gain or gain index approach.

* Decision Tree : In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. This algorithm uses a tree-like model of decisions. This algorithm in data mining helps in deriving a strategy to reach a particular goal. This is also widely used in machine learning.

* We have **implemented Pipelining** with (MinMaxScaler + LogisticRegression) combination and (StandardScaler + LogisticRegression) combination to automate the standard Machine Learning tasks and explore different combinations of transformers and estimators and their performance on the given dataset.

* We have **implemented ensemble methods** with Decision tree and Random forest to make best prediction (whether the client will subscribe for term deposit or not) by aggregating the obtained results.


### Evaluation:
 After understanding the sentiments of the Bank Marketing Data Set. We need to mine knowledge and capture the ideas from the dataset. In this process of capturing the insights from the data, we need to evaluate the results at each data mining phase which is a challenging task.
The common problem which everyone can face is that for unbalanced Bank Marketing Data Set data streams with, for example, 90%
of the instances in one class, the simplest classifiers will have high accuracies of at least 90%.
Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce error. In the Bank-Marketing-Analysis dataset, we have unbalanced data for the target variable 'y'. So, we have balanced it by applying OverSampling technique.

We can use different evaluation techniques like:

•Accuracy

•Confusion matrix

•F1 score

•Precision-Recall or PR curve

•ROC (Receiver Operating Characteristics) curve

**Accuracy:** The Accuracy is the most commonly used metric to judge a data model and is actually not a clear indicator of the performance. The worse happens when classes are imbalanced.
It is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model. For our model, we have got 0.803 which means our model is approx. 80% accurate.

Accuracy = TP+TN/TP+FP+FN+TN

**Confusion Matrix:** The confusion matrix is the summary of prediction results on a classification problem.
The number of correct and incorrect predictions are summarized with count values and are broken down by each class. This is the key to the confusion matrix.
The confusion matrix shows the ways in which the classification model is confused when it makes the predictions.
It gives us the insight not only into errors being made by a classifier but more importantly the types of errors that are being made. So that we can remove them easily.

**Precision-Recall or PR curve:** Precision-Recall is the required measure of success of prediction when the classes are very imbalanced. In information retrieval, precision is the measure of result relevancy, while recall is a measure of how many truly relevant results are returned.
The precision-recall curve shows the tradeoff between precision and recall for different threshold. The high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
A system with the high recall but having low precision returns many results, but most of its predicted labels are incorrect when compared to the training labels. A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. The ideal system with high precision and high recall will return many of the results, with all the results retrieved correctly.



**ROC (Receiver Operating Characteristics) curve:**  The ROC curve is the plot between the true positive rate (TPR) and the false positive rate (FPR) at various threshold settings. The true-positive rate is also known as sensitivity, recall or probability of detection in the process of machine learning. The false-positive rate is also known as the fall-out or probability of false data. It can also be thought of as a plot of the power as a function of the Type I Error of the decision rule. The ROC curve is thus the sensitivity as a function of fall-out. In general methodology, if the probability distributions for both detection and false data are known, the ROC curve can be generated by plotting the cumulative distribution function of the detection probability in the y-axis against the cumulative distribution function of the false-data probability on the x-axis.

### Conclusion:

We have implemented **Random Forest Classifier (classification model)** that can predict accurately whether the client will subscribe term deposit or not.


### Contributors:
1. Aniruddha Sudhindra Shirahatti
2. Chandrakanth Rajesh
3. Krishna Vishwanatham
4. Digvijay Gole
5. Yash Bonde


### Contact
Aniruddha Sudhindra Shirahatti - ashiraha@uncc.edu

Chandrakanth Rajesh - crajesh@uncc.edu

Krishna Vishwanatham - kvishwa1@uncc.edu

Digvijay Gole - dgole@uncc.edu

Yash Bonde - ybonde1@uncc.edu
