# Transparent Unfairness - How to Use the Functions from fairdetect_functions.py
## An approach to detecting and understanding machine learning bias.

Congregating the various theoretical concepts into a practical framework, we can follow the “theoretical lens of a ‘sense-plan-act’ cycle”, as described by the HLEG framework (European Commission and Directorate-General for Communications Networks, Content and Technology, 2019). Applying this concept to the problem of ML fairness, we can break down three core steps in providing robust, and responsible artificial intelligence: Identify, Understand, and Act (IUA).

1. __Identify__: The process of exposing direct or indirect biases within a dataset and/or model.
1. __Understand__: The process of isolating impactful scenarios and obtaining trans parent explanations for outcomes.
1. __Act__: The process of reporting and rectifying identified disparities within the


By understanding the philosophical forms of unfairness as defined by our review of the literature and categorizing our prominent fairness metrics into the overarching categories of representation, ability, and performance, we can establish a series of tests to “identify” levels of disparities between sensitive groups at different levels. Merging these findings with the explainability of our models through the use of white-box models, or Shapley value estimation for black-box models, we can dig deeper into the model’s predictions, “understanding” how classifications were made, and how they varied from the natural dataset exposing both natural biases as well as added model differences. Finally, by probing further into levels of misclassification, in particular looking at negative outcomes, we can isolate groups most at risk and set up a series of “actions” that can be taken to mitigate the effects. Given this three-step framework which combines societal, legal, and technical considerations, the paper will then go through a series of cases, and examine the proposed framework.

# Let's use fairdetect_functions to find biases
For using fairdetect_functions we need create a model, for this we will:
1. __load the functions__ from fairdetect_functions
1. then __load a dataset__, in this case we will use: german dataser from https://dalex.drwhy.ai/python/api/datasets/index.html
/  Dataset 'german' contains information about people and their credit risk. On the base of age, purpose, credit amount, job, sex, etc. the model should predict the target - risk. risk tells if the credit rate will be good (1) or bad (0). This data contains some bias and it can be detected using the dalex.fairness module.
1. then __clean and prepare the dataset__ for modeling, in this case we only need to apply label enconding in order to transform text into numeric variables.
1. then we __split the data in train and test sets__
1. then we __build a model__, in this case we applying a Machine Learning (ML) model called XGBoost Classifier. The reason for choosing a ML is that ML models are harded to visualize, sometimes called Black Box models, what makes very difficult to see if any bias is present in the model.



```python
!pip install dalex
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: dalex in c:\users\manoe\appdata\roaming\python\python39\site-packages (1.4.1)
    Requirement already satisfied: numpy>=1.18.4 in c:\users\manoe\appdata\roaming\python\python39\site-packages (from dalex) (1.21.4)
    Requirement already satisfied: plotly>=4.12.0 in c:\anaconda3\lib\site-packages (from dalex) (5.6.0)
    Requirement already satisfied: pandas>=1.1.2 in c:\users\manoe\appdata\roaming\python\python39\site-packages (from dalex) (1.4.2)
    Requirement already satisfied: scipy>=1.5.4 in c:\anaconda3\lib\site-packages (from dalex) (1.7.3)
    Requirement already satisfied: tqdm>=4.48.2 in c:\users\manoe\appdata\roaming\python\python39\site-packages (from dalex) (4.64.0)
    Requirement already satisfied: setuptools in c:\anaconda3\lib\site-packages (from dalex) (61.2.0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\manoe\appdata\roaming\python\python39\site-packages (from pandas>=1.1.2->dalex) (2022.1)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\manoe\appdata\roaming\python\python39\site-packages (from pandas>=1.1.2->dalex) (2.8.2)
    Requirement already satisfied: six in c:\users\manoe\appdata\roaming\python\python39\site-packages (from plotly>=4.12.0->dalex) (1.16.0)
    Requirement already satisfied: tenacity>=6.2.0 in c:\anaconda3\lib\site-packages (from plotly>=4.12.0->dalex) (8.0.1)
    Requirement already satisfied: colorama in c:\users\manoe\appdata\roaming\python\python39\site-packages (from tqdm>=4.48.2->dalex) (0.4.4)
    


```python
import dalex as dx
import pandas as pd
data = dx.datasets.load_german()
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>risk</th>
      <th>sex</th>
      <th>job</th>
      <th>housing</th>
      <th>saving_accounts</th>
      <th>checking_account</th>
      <th>credit_amount</th>
      <th>duration</th>
      <th>purpose</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>not_known</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>not_known</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>53</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>not_known</td>
      <td>1736</td>
      <td>12</td>
      <td>furniture/equipment</td>
      <td>31</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>little</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
      <td>40</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>not_known</td>
      <td>804</td>
      <td>12</td>
      <td>radio/TV</td>
      <td>38</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
      <td>23</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>moderate</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 10 columns</p>
</div>



# Data Cleaning
## Transforming/coding text into numbers


```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['housing'] = le.fit_transform(data['housing'])
data['saving_accounts'] = le.fit_transform(data['saving_accounts'])
data['checking_account'] = le.fit_transform(data['checking_account'])
data['purpose'] = le.fit_transform(data['purpose'])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>risk</th>
      <th>sex</th>
      <th>job</th>
      <th>housing</th>
      <th>saving_accounts</th>
      <th>checking_account</th>
      <th>credit_amount</th>
      <th>duration</th>
      <th>purpose</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1169</td>
      <td>6</td>
      <td>5</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5951</td>
      <td>48</td>
      <td>5</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2096</td>
      <td>12</td>
      <td>3</td>
      <td>49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7882</td>
      <td>42</td>
      <td>4</td>
      <td>45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4870</td>
      <td>24</td>
      <td>1</td>
      <td>53</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1736</td>
      <td>12</td>
      <td>4</td>
      <td>31</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3857</td>
      <td>30</td>
      <td>1</td>
      <td>40</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>804</td>
      <td>12</td>
      <td>5</td>
      <td>38</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1845</td>
      <td>45</td>
      <td>5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4576</td>
      <td>45</td>
      <td>1</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 10 columns</p>
</div>



## Splitting the data into train and test


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
X = data.drop(["risk"],axis=1) # axis: {0 or ‘index’, 1 or ‘columns’}, default 0
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=0)
print("Data sucessfully loaded!")
```

    Data sucessfully loaded!
    

## Trainning a machine learning model


```python
!pip install xgboost
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: xgboost in c:\users\manoe\appdata\roaming\python\python39\site-packages (1.6.1)
    Requirement already satisfied: scipy in c:\anaconda3\lib\site-packages (from xgboost) (1.7.3)
    Requirement already satisfied: numpy in c:\users\manoe\appdata\roaming\python\python39\site-packages (from xgboost) (1.21.4)
    


```python
import xgboost
model = xgboost.XGBClassifier().fit(X_train, y_train)
y_test_predict = model.predict(X_test)
y_test_predict
```




    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
           1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
           0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
           1, 1])



## Using FairDetect to test biases in the 'sex' variable

```python
!pip install fairdetect
```



```python
from fairdetect import FairDetect
```


```python
fd = FairDetect(model,X_test,y_test)
```


```python
sensitive = 'sex' # sensible variable one wants to test for biases
labels = {0:'Female',1:'Male'} # 0 - Female and 1 - Male
fd.identify_bias(sensitive,labels)
```



    Accept H0: No Significant Predictive Disparity. p= 0.786326136425535
    

In our case study of the German Credit Risk dataset, we are provided with nine, fact-based features to determine whether a candidate is a high credit risk. Among this dataset are two main sensitive groups, sex, and age. For this particular analysis, we will focus on sex as being our sensitive group. To generate predictions, we will use an XGBoost classifier which received an overall test accuracy of (76%) indicating moderate predictive performance.

we will begin by observing representation factors by firstly looking at the representation of both the sex labels and outcome labels. We can see that there are a lot more males than females in the dataset, and in general, more people are labeled as being of high credit risk. This may indicate that our model will have more opportunities to train on such subgroups.

REPRESENTATION

<table>
    
   <tr>
       <td></td>
       <td>Female</td>
       <td>Male</td>
    </tr>
   <tr>
       <td>0</td>
       <td>0.347222</td>
       <td>0.25781</td>
    </tr>
   <tr>
       <td>1</td>
       <td>0.652778</td>
       <td>0.74218</td>
    </tr>    
</table>

Finally, testing for demographic parity, we look at the normalized contingency table between our sex categories and the associated predicted risk, running our chisquared assessment we obtain a p of .823, accepting our null hypothesis of independence and indicating no significant relation between the sex and the risk classification. In addition, running the same analysis on our base dataset provides us with a p of .239, once again indicating no major dependencies of the sensitive variable on the target label. Moving then into the principle of ability, we compute our true-positive and false-positive rates.

* True Positive Rate for Males 85.26%
* True Positive Rate for Females 95.74%

Based on our true positive rates, it seems as though we have a small difference be tween sensitive groups, however, running our chi-squared test, we obtain a p of .436, accepting our null, and indicating no significant difference between our true positive rates. This would thus satisfy equal opportunity for both males and females in which both genders have similar high-risk probabilities given that they in fact present a high risk.

* False Positive Rate for Males 57.58%
* False Positive Rate for Females 52.00%

In addition, the false-positive rates for both groups seem rather consistent with a p of .594, allowing us to accept our null hypothesis of no significant differences between the false-positive rates and noting similar levels of high-risk misclassifications for either group.

* True Negative Rate for Males 42.42%
* True Negative Rate for Females 48.00%

Looking at true negative rates in which our model correctly predicted proper low risk classifications we see similar scores for both groups supported by a p of .558, indicating similar levels of proper low-risk classifications for each gender.

* False Negative Rate for Males: 14.73%
* False Negative Rate for Females: 4.25%

Finally looking at false-negative rates we get a different picture, it seems as though males are being incorrectly labeled as being of low risk at a much higher rate than females, supporting this is a p of .016, while this may be insignificant for alpha levels of 10% and 5% it does not satisfy at 1% indicating a potential advantage for men receiving more credit in cases when they should not be, especially when compared to their female counterparts who are not given this privilege.

* Precision for Men: 81.00%
* Precision for Women: 77.59%


Finally, to observe model exacerbation of biases through the lens of predictive parity, 1021 we notice slight differences in the precision scores for both groups. Looking at the chisquared test, however, we obtain a p of .786 signaling low significance of disparity, we are informed that the model is not greatly hindering either group, but rather enforcing  existing disparities. This is complimentary of our previous analysis in which we only hinted at a slight significant disparity among the false-negative rates.

Having understood the basis of the slight disparity detected, we can isolate the most affected group as being males that are incorrectly classified as having low credit risk. To further investigate, we must dive into the workings of the black-box model. To do so, we will introduce the idea of SHAP through the SHAP library. Our first
result is to observe key feature importance based on our sex cohorts.

## Using FairDetect to retrieve the SHapley Additive exPlanations
SHapley Additive exPlanations (SHAP): The Shapley value is a solution concept in cooperative game theory. It was named in honor of Lloyd Shapley, who introduced it in 1951 and won the Nobel Memorial Prize in Economic Sciences for it in 2012. The objective of this value is to provide the marginal contribution an individual has to a coalition’s output. By observing all the different ways we can compose this coalition, through the inclusion and exclusion of members, and finding the differences in outputs based on individual presence, the shapely value gives us the contribution each member has provided (Roth, 1988). SHapley Additive exPlanations (SHAP) apply this ideology to machine learning models, relating the individuals to features in a dataset, and the coalition output as the predicted value of the model (Lundberg and Lee, 2017).


```python
!pip install shap
```
    


```python
sensitive = 'sex' # sensible variable one wants to test for biases
labels = {0:'Female',1:'Male'} # 0 - Female and 1 - Male
fd.understand_shap(labels,'sex',1,0)
```

    ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    

    Model Importance Comparison
    

    ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    

    


    Affected Attribute Comparison
    Average Comparison to True Class Members
    



    Random Affected Decision Process
    

    ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    


      


Observing the graph on the left, we can see the overall feature importance, and on the right, we see the importance of the isolated male, false-negative cases. In general, it seems as though the same factors are considered at roughly the same weight between genders. However, we do notice a larger gap within duration seeing that it is of higher importance to the model with males. Comparing the graph to the isolated FN cases, we immediately notice a big difference in importance between the credit amount observed in the overall outputs as opposed to the subgroup. It is a clear indication that a certain sub characteristic of the male FN group is contributing to the incorrect labels.

Selecting a random case, we can immediately see our hypothesis in play, as the credit amount of the individual had a significant influence on the model’s decision to predict the candidate as having low risk. However, noticing their checking account label of 0, and cross-checking it with our dataset original labels, we are aware of the candidate being labeled as having a “little” checking account.

Looking at another random case we again see a similar picture. Although the candidate has a very small checking account, the credit amount and duration pushed the model towards a low-credit risk classification.

Finally, by looking at the differences in the average values of the three primary model factors, we can see that the average male, incorrectly labeled as being of low risk tends to have lower checking account labels, a higher credit amount, as well as duration as opposed to people who were correctly labeled as having no risk. Moving into the final “act” stage of our ramework, we can hypothesize that certain men, with small checking accounts, but with higher than average credit amounts and durations could be subject to further inspection to minimize the privilege and minimize risk for the bank. Finally, to culminate the research results, we go into our final case of the synthetic credit card approval dataset in which we will use a Neural Network to cover another model and provide a different view of opaqueness.


```python
import dalex as dx
import pandas as pd
data = dx.datasets.load_german()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['housing'] = le.fit_transform(data['housing'])
data['saving_accounts'] = le.fit_transform(data['saving_accounts'])
data['checking_account'] = le.fit_transform(data['checking_account'])
data['purpose'] = le.fit_transform(data['purpose'])

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
X = data.drop(["risk"],axis=1) # axis: {0 or ‘index’, 1 or ‘columns’}, default 0
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=0)
print("Data sucessfully loaded!")

import xgboost
model = xgboost.XGBClassifier().fit(X_train, y_train)
y_test_predict = model.predict(X_test)
y_test_predict
```

    Data sucessfully loaded!
    




    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
           1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
           0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
           1, 1])



Trained model stored inside FairDected object:


```python
fd.model
```




    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                  missing=nan, monotone_constraints='()', n_estimators=100,
                  n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, reg_lambda=1, ...)


