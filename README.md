## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Devloped by : ANUBHARATHI SS
Reg No : 212223040017
```

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/4edb1355-4b72-4ce9-b63d-882d6d9a1b7f)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/a4df77f2-1e7e-4657-89c2-ef292d17d541)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/261efc3b-b019-4bdd-bca9-1645fe4a08f0)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/42de6a88-e0f6-425d-a002-abcbf2247ada)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/user-attachments/assets/f30f8a2a-9f67-40e3-95a5-5c7efe57abc5)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/5e4df103-d83f-4e59-b31a-7103bed9b68f)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/00cc8a67-5b61-474f-bc75-ce21a8100b76)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/a1056047-3a4f-4b98-bc5f-b9e7459c95ef)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/28e1de22-6959-49b4-a6ab-2e3f43a95d0c)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/8f61044c-8cdc-4f52-8f09-62593127efbf)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/45e15627-d3ce-444e-94cd-f1ea39aff6d5)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/dd8760e4-a81c-4192-a5a4-c58e378532dd)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/4d0882cd-2e16-4331-802a-205c6d049c9a)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4edbe1e2-0a5f-415a-9733-f7a8e1b272f8)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/81792253-adae-4529-ac3a-770b7a78d745)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e0865d58-1eb9-4100-bc9b-c140cbbbb328)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/6f68d3f7-8423-4a25-a8a2-e76abf5e3f14)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/0b5060e1-7076-4247-b4b9-8daf3d501770)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/user-attachments/assets/14a93379-807d-4a11-8447-d6a196e1785e)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/628444f7-8471-4ef0-a1fb-ed7ddcacf48d)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4a1ded16-c3bc-479a-b34f-6cd2c3bc23ad)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f40b2972-92e5-43b3-b45e-944dfa7f3b5a)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9877a1d9-937a-4334-b630-508f7efb5dd8)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f8bf6410-1a8a-48c6-8f32-b29b8d7ef9d0)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/42f2d1e8-cc1a-4681-a552-4386038c44a6)



       
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
      
      
     

       
