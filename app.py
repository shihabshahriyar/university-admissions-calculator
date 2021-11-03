import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv('data.csv')

#all the necessary inputs
GREScore = input('What is your GRE Score? ')
TOEFLScore = input('What is your TOEFL Score? ')
UniversityRanking = input('What is your University Ranking (on a scale of 1-5)? ')
SOP = input('Rate your Statement of Purpose (on a scale of 1-5)? ')
LOP = input('Rate your Letter of Recommendations (on a scale of 1-5)? ')
CGPA = input('What is your CGPA Score (on a scale of 1-10)? ')
Research = input('Did you research properly about your university? Type (1-Yes, 0-No) - ')


X = data.drop(columns=['Chance of Admit ', 'Serial No.']) #test scores and other parameters
y = data['Chance of Admit '] #probability of admission

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe = make_pipeline(StandardScaler(), LinearRegression())

model = pipe.fit(X_train, y_train)

# error = mean_squared_error(y_test, model.predict(X_test))
# score = model.score(X_test, y_test)

results = model.predict([[GREScore, TOEFLScore, UniversityRanking, SOP, LOP, CGPA, Research]])

print(f"You have a {results[0] * 100} chance of getting into this university")
