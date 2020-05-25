# Supervised Learning
## Project: Finding Donors for CharityML


### Data

The modified census dataset consists of approximately 42,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)

### Steps taken:
1. Importing datasets( 'census.csv' and 'test_censis.csv')
2. Explore datasets.
3. Explore target variable.
4. Split dataframe into features and target variable(income)
5. Explore continuous(skewed and non-skewed) and categorical features in dataset.
6. Visualization continuous features-
- Skewed(Skewed is the one, where, 2 halves of the visualisation does not appears as mirror images of each other), or
- Non-Skewed( Non-skewed or Symmetric features is one, where 2 halves appear as mirror images of each other.) 
10. Transforming features-
- Skewed Continuous features-(via Log Tranformation)
- Continuous Features- (Normalization)
- Categorical Features- (One Hot Encoding)
11. Visualisation- Correlation Matrix
12. Data Modeling-
-   Split Dataset 'census' into 'train' and 'Validation'
-   Base Model Metric Calculation
-   Apply machine learning models- 
      Random Forest Classifier
      Gradient Boosting Classifier
      AdaBoost Classifier
      Logistic Regression
      XGBoost Classifier
-   Comparison of models based upon ROC-AUC Score Metrics-
-   Hyperparameter Tuning for AdaBoost Classifier
-   Hyperparameter Tuning for Gradient Boosting Classifier
-   Hyperparameter Tuning for XGBoost Classifier
-   Hyperparameter Tuning for Random Forest Classifier
-   Hyperparameter Tuning for Logistic Regression
-   Look for reduced dataset(with most important features) against full dataset
-   Comparison of all combination of models to choose the best optimized model(Top3 models- AdaBoost, XGBoost and Gradient Boosting).
13. Test dataset-
-   Data Preprocessing and EDA steps, same as 'census'- training dataset
-   Predict model, based upon best optimized model
-   Save and Uploaded Dataset on Kaggle.
