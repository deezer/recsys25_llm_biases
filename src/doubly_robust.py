import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 


#get the data and preprocess
df = pd.read_csv('../../../data/ceph/analysis/bmassonisguerra/rec_llm/rating_cleaned.csv')
user = pd.read_csv('../../../data/ceph/analysis/bmassonisguerra/rec_llm/user_info.csv')
user_stream = pd.read_csv('../../../data/ceph/analysis/bmassonisguerra/rec_llm/user_stream_data.csv')
age_gender = pd.read_csv('../../../data/ceph/analysis/bmassonisguerra/rec_llm/gender-age.csv')

df = df[df['rating'].notna()]

temp = df[df.is_own_profile == False].groupby('user_id')['model'].count().reset_index(name = 'n')
users_remove = temp[temp.n !=5].user_id.values

df = df[~(df.user_id.isin(users_remove))]

temp = df[df.is_own_profile == True].groupby('user_id')['rating'].count().reset_index(name = 'n')
users_keep = temp[temp.n == 12].user_id.values

df = df[df.user_id.isin(users_keep)]

temp = df[df.is_own_profile == False].groupby('user_id')['rating'].median().reset_index(name = 'baseline')
df = df.merge(temp, on = 'user_id')
df['delta_rating'] = df['rating'] - df['baseline']

df = df[df.is_own_profile == True].copy()

df = df.merge(age_gender, on = 'user_id', how = 'left')
df = df.merge(user, on = 'user_id', how = 'left')
df = df.merge(user_stream, on = ['user_id','time_window'], how = 'left')

df[df.filter(like='_ratio').columns] = df.filter(like='_ratio').fillna(0)

df = df.dropna().copy()




df = df[(df.is_own_profile == True)].dropna(subset=["rating"])
df.reset_index(inplace = True, drop = True)

df.rename(columns=lambda x: x.replace("_ratio", "") if "_ratio" in x else x, inplace=True)

#get long term consumption for covariates 
df_prop = pd.read_csv('../../../data/nfs/analysis/bmassonisguerra/rec_llm/propensity.csv')
df_prop = df_prop[['user_id','Electronic_ratio', 'Metal_ratio', 'Rock_ratio', 'Jazz_ratio','World_ratio','Rap_ratio']].copy()


temp = df[['user_id','delta_rating','model',
           'gs_score','age','Electronic', 
           'Metal', 'Rock', 'United States of America', 'United Kingdom', 'Canada', 'Jazz', 'France',
           'Belgium', 'World', 'Rap',
           'genre_entropy','genre_none', 'country_entropy', 'country_none']].copy()

temp = temp.merge(df_prop, on = 'user_id')
temp['US'] = temp['United States of America']
temp['UK'] = temp['United Kingdom']

temp = temp.dropna()

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

covariates = ['gs_score','age','Electronic', 
              'Metal', 'Rock', 'US', 'UK', 'Canada', 'Jazz', 'France',
              'Belgium', 'World', 'Rap', 
              'genre_entropy','genre_none', 'country_entropy', 'country_none',
              'Electronic_ratio', 'Metal_ratio', 'Rock_ratio', 'Jazz_ratio','World_ratio','Rap_ratio'
             ]



#run bootstrap 
results = []

for param in tqdm(['Metal', 'Rock', 'US', 'UK', 'Canada', 'Jazz', 'France',
           'Belgium', 'World', 'Electronic', 'Rap']):
    
    X = [item for item in covariates if item != param ]
    
    for model in df.model.unique():   
        data = temp[temp.model == model].copy()
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

df_ate.to_csv('dobly_robustCR.csv', index=False)
