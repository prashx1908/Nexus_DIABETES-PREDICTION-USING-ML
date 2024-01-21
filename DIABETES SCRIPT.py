# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.201978Z","iopub.execute_input":"2024-01-20T06:35:02.202382Z","iopub.status.idle":"2024-01-20T06:35:02.207959Z","shell.execute_reply.started":"2024-01-20T06:35:02.202353Z","shell.execute_reply":"2024-01-20T06:35:02.206730Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.210206Z","iopub.execute_input":"2024-01-20T06:35:02.210673Z","iopub.status.idle":"2024-01-20T06:35:02.228351Z","shell.execute_reply.started":"2024-01-20T06:35:02.210629Z","shell.execute_reply":"2024-01-20T06:35:02.227178Z"},"jupyter":{"outputs_hidden":false}}
df=pd.read_csv('/kaggle/input/diabetic-data-set-for-disease-prediction/diabetes.csv')

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.230176Z","iopub.execute_input":"2024-01-20T06:35:02.230603Z","iopub.status.idle":"2024-01-20T06:35:02.253632Z","shell.execute_reply.started":"2024-01-20T06:35:02.230557Z","shell.execute_reply":"2024-01-20T06:35:02.252432Z"},"jupyter":{"outputs_hidden":false}}
df

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.254956Z","iopub.execute_input":"2024-01-20T06:35:02.255832Z","iopub.status.idle":"2024-01-20T06:35:02.268692Z","shell.execute_reply.started":"2024-01-20T06:35:02.255790Z","shell.execute_reply":"2024-01-20T06:35:02.267780Z"},"jupyter":{"outputs_hidden":false}}
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.272299Z","iopub.execute_input":"2024-01-20T06:35:02.272695Z","iopub.status.idle":"2024-01-20T06:35:02.284269Z","shell.execute_reply.started":"2024-01-20T06:35:02.272663Z","shell.execute_reply":"2024-01-20T06:35:02.282893Z"},"jupyter":{"outputs_hidden":false}}
print(df.duplicated())
print(df.duplicated().sum())

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.285662Z","iopub.execute_input":"2024-01-20T06:35:02.286194Z","iopub.status.idle":"2024-01-20T06:35:02.292423Z","shell.execute_reply.started":"2024-01-20T06:35:02.286163Z","shell.execute_reply":"2024-01-20T06:35:02.291293Z"},"jupyter":{"outputs_hidden":false}}
df.drop_duplicates(inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.294057Z","iopub.execute_input":"2024-01-20T06:35:02.294368Z","iopub.status.idle":"2024-01-20T06:35:02.305055Z","shell.execute_reply.started":"2024-01-20T06:35:02.294342Z","shell.execute_reply":"2024-01-20T06:35:02.303905Z"},"jupyter":{"outputs_hidden":false}}
df.duplicated().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.306799Z","iopub.execute_input":"2024-01-20T06:35:02.307227Z","iopub.status.idle":"2024-01-20T06:35:02.315532Z","shell.execute_reply.started":"2024-01-20T06:35:02.307184Z","shell.execute_reply":"2024-01-20T06:35:02.314362Z"},"jupyter":{"outputs_hidden":false}}
df.isna().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.317107Z","iopub.execute_input":"2024-01-20T06:35:02.317444Z","iopub.status.idle":"2024-01-20T06:35:02.323448Z","shell.execute_reply.started":"2024-01-20T06:35:02.317415Z","shell.execute_reply":"2024-01-20T06:35:02.322382Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.preprocessing import StandardScaler

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.324400Z","iopub.execute_input":"2024-01-20T06:35:02.324732Z","iopub.status.idle":"2024-01-20T06:35:02.340695Z","shell.execute_reply.started":"2024-01-20T06:35:02.324705Z","shell.execute_reply":"2024-01-20T06:35:02.339521Z"},"jupyter":{"outputs_hidden":false}}
features=['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
scaler = StandardScaler()
df[features]= scaler.fit_transform(df[features])

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.344369Z","iopub.execute_input":"2024-01-20T06:35:02.344753Z","iopub.status.idle":"2024-01-20T06:35:02.368765Z","shell.execute_reply.started":"2024-01-20T06:35:02.344722Z","shell.execute_reply":"2024-01-20T06:35:02.367645Z"},"jupyter":{"outputs_hidden":false}}
df

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.370179Z","iopub.execute_input":"2024-01-20T06:35:02.370945Z","iopub.status.idle":"2024-01-20T06:35:02.380869Z","shell.execute_reply.started":"2024-01-20T06:35:02.370905Z","shell.execute_reply":"2024-01-20T06:35:02.379788Z"},"jupyter":{"outputs_hidden":false}}
X= df[features]
y=df['Outcome']
X_train,X_test, y_train, y_test= train_test_split(X,y, test_size= 0.2, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.382937Z","iopub.execute_input":"2024-01-20T06:35:02.383425Z","iopub.status.idle":"2024-01-20T06:35:02.405563Z","shell.execute_reply.started":"2024-01-20T06:35:02.383384Z","shell.execute_reply":"2024-01-20T06:35:02.404493Z"},"jupyter":{"outputs_hidden":false}}
df

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.406971Z","iopub.execute_input":"2024-01-20T06:35:02.407816Z","iopub.status.idle":"2024-01-20T06:35:02.414674Z","shell.execute_reply.started":"2024-01-20T06:35:02.407783Z","shell.execute_reply":"2024-01-20T06:35:02.413535Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.linear_model import LogisticRegression

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.416113Z","iopub.execute_input":"2024-01-20T06:35:02.416545Z","iopub.status.idle":"2024-01-20T06:35:02.435988Z","shell.execute_reply.started":"2024-01-20T06:35:02.416515Z","shell.execute_reply":"2024-01-20T06:35:02.434724Z"},"jupyter":{"outputs_hidden":false}}
model= LogisticRegression()
model.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.437533Z","iopub.execute_input":"2024-01-20T06:35:02.438609Z","iopub.status.idle":"2024-01-20T06:35:02.443295Z","shell.execute_reply.started":"2024-01-20T06:35:02.438570Z","shell.execute_reply":"2024-01-20T06:35:02.442061Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.444801Z","iopub.execute_input":"2024-01-20T06:35:02.445126Z","iopub.status.idle":"2024-01-20T06:35:02.463476Z","shell.execute_reply.started":"2024-01-20T06:35:02.445098Z","shell.execute_reply":"2024-01-20T06:35:02.462703Z"},"jupyter":{"outputs_hidden":false}}
y_pred= model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:35:02.464993Z","iopub.execute_input":"2024-01-20T06:35:02.465523Z","iopub.status.idle":"2024-01-20T06:35:02.472531Z","shell.execute_reply.started":"2024-01-20T06:35:02.465493Z","shell.execute_reply":"2024-01-20T06:35:02.471484Z"},"jupyter":{"outputs_hidden":false}}
new_predictions = model.predict(new_data)


print("Predictions:", new_predictions)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-20T06:36:01.627364Z","iopub.execute_input":"2024-01-20T06:36:01.627910Z","iopub.status.idle":"2024-01-20T06:36:36.402846Z","shell.execute_reply.started":"2024-01-20T06:36:01.627774Z","shell.execute_reply":"2024-01-20T06:36:36.401692Z"},"jupyter":{"outputs_hidden":false}}

user_input = {}

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for feature in feature_names:
    user_input[feature] = float(input(f"Enter the value for {feature}: "))
new_data = pd.DataFrame(user_input, index=[0], columns=feature_names)

new_data_standardized = scaler.transform(new_data)
new_predictions = model.predict(new_data_standardized)


print("Predictions:", new_predictions)