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
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/cbdffb8c-3a86-48ee-b766-a468ad5cbd49)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
 pm=['Hot','Warm','Cold']
 e1=OrdinalEncoder(categories=[pm])
 e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/ed9f0ed1-2a49-4fc2-9555-2b2bf8eb638a)

 ```
df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
```
![image](https://github.com/user-attachments/assets/4543142d-8666-48e0-b7e8-bc6e0eea4c42)
```
le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
![image](https://github.com/user-attachments/assets/3a3142b9-83a7-4a37-a37c-31d3d899d143)
```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
df2 = df.copy() 
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
![image](https://github.com/user-attachments/assets/f8d807b7-e4eb-4051-a1b0-ee7cf5e4ee48)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/c6f78e52-3889-494e-a008-a4045ee159c9)

```
!pip install category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/3a3def1f-a9f9-468e-80f2-532299f33b3c)
```
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 df
```
![image](https://github.com/user-attachments/assets/287afcfc-f8b1-4732-a01f-b6042c9e4d73)

```
dfb=pd.concat([df,nd],axis=1)
 dfb
```
![image](https://github.com/user-attachments/assets/8e43fe20-135d-40ab-80bc-252bdf989323)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/1400a965-a651-4f2f-adf2-1da6da6220ca)

```
import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
![image](https://github.com/user-attachments/assets/e4255512-f20c-4faf-9085-4b12a1d9de67)

```
 df.skew()
```
![image](https://github.com/user-attachments/assets/4dcbdd91-28a7-416e-822d-64a5a84e475c)

```
 np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1fa6549e-e0cf-40de-b537-43802df574a1)

```
 np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/41e79e15-ba77-4d0b-800e-a750aeadfdbb)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f2421ffc-86ee-434a-ac23-835b8ac2e7d9)
```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
![image](https://github.com/user-attachments/assets/30a7c5fb-5742-4c60-87a3-60f1466585dc)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/c501b32f-3479-4604-b9d8-063e61e33054)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/ddc10e78-baa4-4fe1-864a-552f1a0ae1ea)

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
![image](https://github.com/user-attachments/assets/7f5ff964-d43e-45d9-b3ac-1dbd5e4454f1)

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/4bcda7fb-93b9-4969-9171-93ec2a73aa2a)
```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/ffd89575-94bb-460c-8fb8-b55361f8535c)
```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/51a9ccad-3818-4cb2-9dac-bca4f2d1ba82)
```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/f4804ef6-00ba-4ce3-900a-c2bf88f4ffa9)
```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/0806dabf-4d68-4a4a-9ea6-a5ac80cb4053)
```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
plt.show()
```
```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()
```

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
