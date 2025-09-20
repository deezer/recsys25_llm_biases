import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 


#get the data and preprocess
df = pd.read_csv('data.csv')
#get the long term preferences
df_prop = pd.read_csv('long_term.csv')

df_prop = df_prop[['user_id','Electronic_ratio', 'Metal_ratio', 'Rock_ratio', 'Jazz_ratio','World_ratio','Rap_ratio']].copy()

temp = df[['user_id','delta_rating','model',
           'gs_score','Electronic', 
           'Metal', 'Rock', 'US', 'UK', 'Canada', 'Jazz', 'France',
           'Belgium', 'World', 'Rap',
           'genre_entropy','genre_none', 'country_entropy', 'country_none']].copy()

temp = temp.merge(df_prop, on = 'user_id')


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

covariates = ['gs_score','Electronic', 
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
        for i in range(10):#1000 times in the paper
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

ate =  pd.DataFrame(results)

ate['plot_model'] = 'llama'
ate.loc[ate.model == 'deepseek-r1', 'plot_model'] = 'deepseek'
ate.loc[ate.model == 'gemini-2.0-flash', 'plot_model'] = 'gemini'

predictor_means = (
    ate.groupby('treat')['mean']
    .mean()
    .sort_values()
    .reset_index()
)

# Assign global order
predictor_order = {p: i for i, p in enumerate(predictor_means['treat'])}
ate['predictor_ordered'] = ate['treat'].map(predictor_order)

g = sns.FacetGrid(
    data=ate,
    col='plot_model',
    sharex=True,
    sharey=True,
    height=3,
    aspect=1.1
)

def plot_bars_with_ci(data, color, **kwargs):
    # Sort inside the plotting function to preserve order per facet
    data = data.sort_values('predictor_ordered')
    y_positions = range(len(data))
    plt.barh(
        y=y_positions,
        width=data['mean'],
        xerr=[data['mean'] - data['ci_lower'], data['ci_upper'] - data['mean']],
        color=color,
        edgecolor='black',
        height=0.6,
        alpha=0.8
    )
    plt.yticks(y_positions, data['treat'])

g.map_dataframe(plot_bars_with_ci, color='cornflowerblue')

# Add vertical line at zero
for ax in g.axes.flat:
    ax.axvline(0, color='red', linestyle='--', linewidth=1)

g.set_titles(col_template="{col_name}")
g.set_axis_labels("Estimated ATE 95% CI")

plt.tight_layout()
plt.savefig("ATE.pdf", bbox_inches='tight')
