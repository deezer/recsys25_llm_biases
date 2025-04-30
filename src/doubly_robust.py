import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 


def doubly_robust(df, X, t, Y):
    df['treatment'] = (df[t] > df[t].median()).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[X])
    
    # Estimate the propensity scores using Logistic Regression with cross-validation for regularization
    ps = LogisticRegression(C=1e-6, max_iter=5000).fit(X_scaled, df['treatment']).predict_proba(X_scaled)[:, 1]
    
    # Fit the outcome models using Ridge Regression (with regularization)
    model0 = GradientBoostingRegressor().fit(df[df.treatment == 0][X], df[df.treatment == 0][Y])
    mu0 = model0.predict(df[X])
    
    model1 = GradientBoostingRegressor().fit(df[df.treatment == 1][X], df[df.treatment == 1][Y])
    mu1 = model1.predict(df[X])
    
    # Doubly Robust Estimator for ATE
    ate = (
        np.mean(df['treatment'] * (df[Y] - mu1) / ps + mu1) -
        np.mean((1 - df['treatment']) * (df[Y] - mu0) / (1 - ps) + mu0)
    )
    
    return ate

covariates = ['Rock', 'Rap', 'World', 'Metal', 'Electronic', 'Classical','genre_none', 
              'Rock_ratio', 'Rap_ratio', 'World_ratio', 'Metal_ratio', 'Electronic_ratio', 'Classical_ratio',
              'genre_none_ratio','genre_entropy','age','gs_score' ]

#get the data
df = pd.read_csv("data.csv")

#run bootstrap 
results = []

for param in tqdm(['Rock', 'Rap', 'World', 'Metal', 'Electronic', 'Classical', 'US', 'France', 'UK']):
    
    X = [item for item in covariates if item != param ]
    
    for model in df.model.unique():   
        data = df[df.model == model].copy()
        print(model)
        ates = []
        for i in range(1000):
            ates.append(doubly_robust(data.sample(frac=1, replace=True), X, param,'delta_rating'))
        print(f"ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))
       
       

        mean = np.mean(ates)
        ci_lower = np.percentile(ates, 2.5)
        ci_upper = np.percentile(ates, 97.5)

        results.append({
            'treat': param, 
            'model': model,
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

df_ate =  pd.DataFrame(results)

df_ate.to_csv('dobly_robust.csv', index=False)
