import pandas as pd
df = pd.read_csv("matches.csv")
df.head(3)

df.drop(columns=["id","method","umpire1","umpire2","result_margin","super_over","target_overs"],inplace=True)
df.head(3)

df.dropna(subset=["winner"],inplace=True)
df["winner"].isnull().sum()

import pandas as pd

# 1. Create a dictionary mapping full team names to short codes
team_mapping = {
    'Mumbai Indians': 'MI',
    'Royal Challengers Bangalore': 'RCB',
    'Royal Challengers Bengaluru': 'RCB', # Handles name change in newer seasons
    'Chennai Super Kings': 'CSK',
    'Kolkata Knight Riders': 'KKR',
    'Delhi Daredevils': 'DC',
    'Delhi Capitals': 'DC',
    'Kings XI Punjab': 'PBKS',
    'Punjab Kings': 'PBKS',
    'Rajasthan Royals': 'RR',
    'Sunrisers Hyderabad': 'SRH',
    'Deccan Chargers': 'HDC',
    'Gujarat Titans': 'GT',
    'Lucknow Super Giants': 'LSG',
    'Rising Pune Supergiant': 'RPS',
    'Rising Pune Supergiants': 'RPS',
    'Kochi Tuskers Kerala': 'KTK',
    'Pune Warriors': 'PWI',
    'Gujarat Lions':'GL'
}

# 2. Map full names to abbreviations across all team columns
team_cols = ['team1', 'team2', 'winner', 'toss_winner']

for col in team_cols:
    if col in df.columns:
        df[col] = df[col].map(team_mapping).fillna(df[col])

import pandas as pd

venue_to_city_map = {
    # Bangalore / Bengaluru
    'M Chinnaswamy Stadium': 'Bangalore',
    'M.Chinnaswamy Stadium': 'Bangalore',
    'M Chinnaswamy Stadium, Bengaluru': 'Bangalore',

    # Mohali / Chandigarh
    'Punjab Cricket Association Stadium, Mohali': 'Mohali',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Mohali',
    'Punjab Cricket Association IS Bindra Stadium': 'Mohali',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Mohali',

    # Delhi
    'Feroz Shah Kotla': 'Delhi',
    'Arun Jaitley Stadium': 'Delhi',
    'Arun Jaitley Stadium, Delhi': 'Delhi',

    # Mumbai
    'Wankhede Stadium': 'Mumbai',
    'Wankhede Stadium, Mumbai': 'Mumbai',
    'Dr DY Patil Sports Academy': 'Mumbai',
    'Dr DY Patil Sports Academy, Mumbai': 'Mumbai',
    'Brabourne Stadium': 'Mumbai',
    'Brabourne Stadium, Mumbai': 'Mumbai',

    # Kolkata
    'Eden Gardens': 'Kolkata',
    'Eden Gardens, Kolkata': 'Kolkata',

    # Jaipur
    'Sawai Mansingh Stadium': 'Jaipur',
    'Sawai Mansingh Stadium, Jaipur': 'Jaipur',

    # Hyderabad
    'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',
    'Rajiv Gandhi International Stadium': 'Hyderabad',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Hyderabad',

    # Chennai
    'MA Chidambaram Stadium, Chepauk': 'Chennai',
    'MA Chidambaram Stadium': 'Chennai',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'Chennai',

    # South African Venues (2009 IPL)
    'Newlands': 'Cape Town',
    "St George's Park": 'Gqeberha',
    'Kingsmead': 'Durban',
    'SuperSport Park': 'Centurion',
    'Buffalo Park': 'East London',
    'New Wanderers Stadium': 'Johannesburg',
    'De Beers Diamond Oval': 'Kimberley',
    'OUTsurance Oval': 'Bloemfontein',

    # Ahmedabad
    'Sardar Patel Stadium, Motera': 'Ahmedabad',
    'Narendra Modi Stadium, Ahmedabad': 'Ahmedabad',

    # Cuttack
    'Barabati Stadium': 'Cuttack',

    # Nagpur
    'Vidarbha Cricket Association Stadium, Jamtha': 'Nagpur',

    # Dharamsala
    'Himachal Pradesh Cricket Association Stadium': 'Dharamsala',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Dharamsala',

    # Kochi
    'Nehru Stadium': 'Kochi',

    # Indore
    'Holkar Cricket Stadium': 'Indore',

    # Visakhapatnam
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Visakhapatnam',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Visakhapatnam',

    # Pune
    'Subrata Roy Sahara Stadium': 'Pune',
    'Maharashtra Cricket Association Stadium': 'Pune',
    'Maharashtra Cricket Association Stadium, Pune': 'Pune',

    # Raipur
    'Shaheed Veer Narayan Singh International Stadium': 'Raipur',

    # Ranchi
    'JSCA International Stadium Complex': 'Ranchi',

    # UAE Venues
    'Sheikh Zayed Stadium': 'Abu Dhabi',
    'Zayed Cricket Stadium, Abu Dhabi': 'Abu Dhabi',
    'Sharjah Cricket Stadium': 'Sharjah',
    'Dubai International Cricket Stadium': 'Dubai',

    # Rajkot
    'Saurashtra Cricket Association Stadium': 'Rajkot',

    # Kanpur
    'Green Park': 'Kanpur',

    # Lucknow
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Lucknow',

    # Guwahati
    'Barsapara Cricket Stadium, Guwahati': 'Guwahati',

    # Mullanpur
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur': 'Mullanpur'
}

# 1. Map stadium strings to clean city names
df['city'] = df['city'].map(venue_to_city_map).fillna(df['city'])

# 2. If 'city' is still null, attempt to map from 'venue' column
if 'venue' in df.columns:
    df['city'] = df['city'].fillna(df['venue'].map(venue_to_city_map))

# 3. Fill any remaining unmapped values
df['city'] = df['city'].fillna('Unknown')

import pandas as pd

# Extract the 4-digit year as an integer
df['season_year'] = df['season'].astype(str).str.extract(r'(\d{4})').astype(int)

df.drop(columns=["season"],inplace=True)
df.head(2)

df['target'] = (df['winner'] == df['team1']).astype(int)
X = df.drop(columns=["winner","target","player_of_match","result","target_runs"])
y = df["target"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

# 1. Split FIRST to prevent data leakage
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Fit encoder strictly on training data
encoder = TargetEncoder(smooth="auto", cv=5)

# 3. Transform city into a single float column representing historical win rate in that city
x_train['city_encoded'] = encoder.fit_transform(x_train[['city']], y_train)
x_test['city_encoded'] = encoder.transform(x_test[['city']])

x_train = x_train.drop(columns=["city"])
x_test = x_test.drop(columns=["city"])

x_train = x_train.drop(columns="venue")
x_test = x_test.drop(columns=["venue"])

x_train

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# --- STEP 1: Feature Engineering (Binary Toss Flag) ---
# Check if team1 won the toss (1) or team2 won the toss (0)
x_train['is_toss_winner_team1'] = (x_train['toss_winner'] == x_train['team1']).astype(int)
x_test['is_toss_winner_team1'] = (x_test['toss_winner'] == x_test['team1']).astype(int)

# Optional: You can drop 'toss_winner' now if 'is_toss_winner_team1' captures it sufficiently
# x_train = x_train.drop(columns=['toss_winner'])
# x_test = x_test.drop(columns=['toss_winner'])


# --- STEP 2: Shared Ordinal Encoding for Team Columns ---
team_cols = ['team1', 'team2', 'toss_winner']

# Initialize a single encoder for all team columns

# Fit encoder on the master list of all unique teams across train columns
all_teams = pd.concat([x_train[col] for col in team_cols]).unique().reshape(-1, 1)
# Fit the encoder on all available team names
team_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
team_encoder.fit(all_teams)

# Transform each team column consistently across train and test sets
for col in team_cols:
    x_train[col] = team_encoder.transform(x_train[[col]])
    x_test[col] = team_encoder.transform(x_test[[col]])

x_train

df["match_type"].unique().tolist()

match_mapped = {
    "League": 0,
    "3rd Place Play-Off": 1,
    "Elimination Final": 2,
    "Eliminator": 2,
    "Qualifier 1": 3,
    "Qualifier 2": 4,
    "Semi Final": 5,
    "Final": 6
}


x_train["match_type_encoded"] = x_train["match_type"].map(match_mapped).astype(int)
x_test["match_type_encoded"] = x_test["match_type"].map(match_mapped).astype(int)

x_train = x_train.drop(columns=["match_type"])
x_test = x_test.drop(columns=["match_type"])

x_train

df["toss_decision"].nunique()


decision = {
    "field":0,
    "bat":1
}
x_train["toss_decision_encoded"] = x_train["toss_decision"].map(decision).astype(int)
x_test["toss_decision_encoded"] = x_test["toss_decision"].map(decision).astype(int)

x_train = x_train.drop(columns={"toss_decision"})
x_test = x_test.drop(columns=["toss_decision"])

import pandas as pd

for df_split in [x_train, x_test]:
    dt_series = pd.to_datetime(df_split['date'])
    df_split['month'] = dt_series.dt.month
    df_split['day_of_week'] = dt_series.dt.dayofweek # 0=Mon, 6=Sun
    df_split['is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)

# Drop original date string column
x_train = x_train.drop(columns=['date'])
x_test = x_test.drop(columns=['date'])

# Verify all features are numeric
print("Train features dtypes:")
print(x_train.dtypes)

# Ensure train and test shapes match
print("\nTrain shape:", x_train.shape)
print("Test shape:", x_test.shape)

x_train

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Baseline Logistic Regression on current x_train
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(x_train)
X_te_s = scaler.transform(x_test)

lr_base = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
lr_base.fit(X_tr_s, y_train)
print("Base LogReg Acc:", accuracy_score(y_test, lr_base.predict(X_te_s)))

# 1. Feature: Team overall win rate in training
team_wins = {}
for t in x_train['team1'].unique():
    t_as_t1 = x_train[x_train['team1'] == t]
    t_as_t2 = x_train[x_train['team2'] == t]
    wins = y_train.loc[t_as_t1.index].sum() + (len(t_as_t2) - y_train.loc[t_as_t2.index].sum())
    total = len(t_as_t1) + len(t_as_t2)
    team_wins[t] = wins / total if total > 0 else 0.5

x_train['t1_win_rate'] = x_train['team1'].map(team_wins).fillna(0.5)
x_test['t1_win_rate'] = x_test['team1'].map(team_wins).fillna(0.5)
x_train['t2_win_rate'] = x_train['team2'].map(team_wins).fillna(0.5)
x_test['t2_win_rate'] = x_test['team2'].map(team_wins).fillna(0.5)
x_train['strength_diff'] = x_train['t1_win_rate'] - x_train['t2_win_rate']
x_test['strength_diff'] = x_test['t1_win_rate'] - x_test['t2_win_rate']

# Batting first
x_train['t1_bat_first'] = ((x_train['is_toss_winner_team1'] == 1) & (x_train['toss_decision_encoded'] == 1)) | ((x_train['is_toss_winner_team1'] == 0) & (x_train['toss_decision_encoded'] == 0))
x_test['t1_bat_first'] = ((x_test['is_toss_winner_team1'] == 1) & (x_test['toss_decision_encoded'] == 1)) | ((x_test['is_toss_winner_team1'] == 0) & (x_test['toss_decision_encoded'] == 0))
x_train['t1_bat_first'] = x_train['t1_bat_first'].astype(int)
x_test['t1_bat_first'] = x_test['t1_bat_first'].astype(int)

# H2H Win Rate between team1 and team2
h2h_wins = {}
for idx, row in x_train.iterrows():
    t1, t2 = row['team1'], row['team2']
    pair = tuple(sorted([t1, t2]))
    if pair not in h2h_wins:
        # compute in train
        m = x_train[((x_train['team1']==t1)&(x_train['team2']==t2)) | ((x_train['team1']==t2)&(x_train['team2']==t1))]
        if len(m) > 0:
            # wins of t1 vs t2
            m_t1_as_t1 = m[m['team1'] == t1]
            m_t1_as_t2 = m[m['team2'] == t1]
            w = y_train.loc[m_t1_as_t1.index].sum() + (len(m_t1_as_t2) - y_train.loc[m_t1_as_t2.index].sum())
            h2h_wins[(t1, t2)] = w / len(m)
        else:
            h2h_wins[(t1, t2)] = 0.5

def get_h2h(row):
    t1, t2 = row['team1'], row['team2']
    return h2h_wins.get((t1, t2), 0.5)

x_train['h2h_t1_win_rate'] = x_train.apply(get_h2h, axis=1)
x_test['h2h_t1_win_rate'] = x_test.apply(get_h2h, axis=1)

best_acc = 0
best_C = None
best_penalty = None

for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    for penalty in ['l1', 'l2']:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(x_train)
        X_te = scaler.transform(x_test)
        
        lr = LogisticRegression(max_iter=2000, C=C, penalty=penalty, solver='liblinear', random_state=42)
        lr.fit(X_tr, y_train)
        acc = accuracy_score(y_test, lr.predict(X_te))
        if acc > best_acc:
            best_acc = acc
            best_C = C
            best_penalty = penalty

print(f"Engineered LogReg Best Acc: {best_acc} (C={best_C}, penalty={best_penalty})")
