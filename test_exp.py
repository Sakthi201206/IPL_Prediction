import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss

# Load data
df = pd.read_csv("matches.csv")
df.dropna(subset=["winner"], inplace=True)

team_mapping = {
    'Mumbai Indians': 'MI',
    'Royal Challengers Bangalore': 'RCB',
    'Royal Challengers Bengaluru': 'RCB',
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

team_cols = ['team1', 'team2', 'winner', 'toss_winner']
for col in team_cols:
    if col in df.columns:
        df[col] = df[col].map(team_mapping).fillna(df[col])

venue_to_city_map = {
    'M Chinnaswamy Stadium': 'Bangalore', 'M.Chinnaswamy Stadium': 'Bangalore', 'M Chinnaswamy Stadium, Bengaluru': 'Bangalore',
    'Punjab Cricket Association Stadium, Mohali': 'Mohali', 'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Mohali',
    'Punjab Cricket Association IS Bindra Stadium': 'Mohali', 'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Mohali',
    'Feroz Shah Kotla': 'Delhi', 'Arun Jaitley Stadium': 'Delhi', 'Arun Jaitley Stadium, Delhi': 'Delhi',
    'Wankhede Stadium': 'Mumbai', 'Wankhede Stadium, Mumbai': 'Mumbai', 'Dr DY Patil Sports Academy': 'Mumbai',
    'Dr DY Patil Sports Academy, Mumbai': 'Mumbai', 'Brabourne Stadium': 'Mumbai', 'Brabourne Stadium, Mumbai': 'Mumbai',
    'Eden Gardens': 'Kolkata', 'Eden Gardens, Kolkata': 'Kolkata',
    'Sawai Mansingh Stadium': 'Jaipur', 'Sawai Mansingh Stadium, Jaipur': 'Jaipur',
    'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad', 'Rajiv Gandhi International Stadium': 'Hyderabad',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Hyderabad',
    'MA Chidambaram Stadium, Chepauk': 'Chennai', 'MA Chidambaram Stadium': 'Chennai', 'MA Chidambaram Stadium, Chepauk, Chennai': 'Chennai',
    'Sardar Patel Stadium, Motera': 'Ahmedabad', 'Narendra Modi Stadium, Ahmedabad': 'Ahmedabad',
    'Maharashtra Cricket Association Stadium': 'Pune', 'Maharashtra Cricket Association Stadium, Pune': 'Pune',
    'Subrata Roy Sahara Stadium': 'Pune',
    'Sheikh Zayed Stadium': 'Abu Dhabi', 'Zayed Cricket Stadium, Abu Dhabi': 'Abu Dhabi',
    'Sharjah Cricket Stadium': 'Sharjah', 'Dubai International Cricket Stadium': 'Dubai',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Lucknow',
}

df['city'] = df['city'].map(venue_to_city_map).fillna(df['city'])
if 'venue' in df.columns:
    df['city'] = df['city'].fillna(df['venue'].map(venue_to_city_map))
df['city'] = df['city'].fillna('Unknown')

df['date'] = pd.to_datetime(df['date'])
df['season_year'] = df['season'].astype(str).str.extract(r'(\d{4})').astype(int)
df = df.sort_values('date').reset_index(drop=True)

# Generate Chronological / Historical Features (No Data Leakage)
all_teams = list(set(df['team1']).union(set(df['team2'])))

# 1. Elo Ratings
elo_ratings = {team: 1500.0 for team in all_teams}
K = 30.0
elo_t1 = []
elo_t2 = []

# 2. H2H Win Rates
h2h_stats = {}

# 3. Recent Form (Last 5 & 10 matches)
team_history = {team: [] for team in all_teams}
t1_w5, t2_w5, t1_w10, t2_w10 = [], [], [], []

# 4. Venue Win Rates
venue_stats = {}
t1_v_rate, t2_v_rate = [], []

# 5. Overall Win Rates
team_overall = {team: [0, 0] for team in all_teams}
t1_ov_rate, t2_ov_rate = [], []

# 6. Season Win Rates
season_team_stats = {}
t1_sea_rate, t2_sea_rate = [], []

# 7. Venue Toss Winner Win Rate
venue_toss_stats = {}
v_toss_winrate = []

h2h_rate = []

current_season = None

for idx, row in df.iterrows():
    t1, t2, w, c, season = row['team1'], row['team2'], row['winner'], row['city'], row['season_year']
    
    # Season reset check
    if season != current_season:
        current_season = season
        season_team_stats = {team: [0, 0] for team in all_teams}
        
    # 1. Elo
    r1 = elo_ratings[t1]
    r2 = elo_ratings[t2]
    elo_t1.append(r1)
    elo_t2.append(r2)
    
    # 2. H2H
    pair = tuple(sorted([t1, t2]))
    if pair not in h2h_stats:
        h2h_stats[pair] = {t1: 0, t2: 0, 'total': 0}
    tot_h2h = h2h_stats[pair]['total']
    w_h2h = h2h_stats[pair][t1]
    h2h_rate.append((w_h2h + 1) / (tot_h2h + 2))
    
    # 3. Form (Last 5 & 10)
    h1 = team_history[t1]
    h2 = team_history[t2]
    t1_w5.append(np.mean(h1[-5:]) if len(h1) > 0 else 0.5)
    t2_w5.append(np.mean(h2[-5:]) if len(h2) > 0 else 0.5)
    t1_w10.append(np.mean(h1[-10:]) if len(h1) > 0 else 0.5)
    t2_w10.append(np.mean(h2[-10:]) if len(h2) > 0 else 0.5)
    
    # 4. Venue
    st1 = venue_stats.get((c, t1), [0, 0])
    st2 = venue_stats.get((c, t2), [0, 0])
    t1_v_rate.append((st1[0] + 1) / (st1[1] + 2))
    t2_v_rate.append((st2[0] + 1) / (st2[1] + 2))
    
    # 5. Overall
    o1 = team_overall[t1]
    o2 = team_overall[t2]
    t1_ov_rate.append((o1[0] + 1) / (o1[1] + 2))
    t2_ov_rate.append((o2[0] + 1) / (o2[1] + 2))
    
    # 6. Season
    sea1 = season_team_stats[t1]
    sea2 = season_team_stats[t2]
    t1_sea_rate.append((sea1[0] + 1) / (sea1[1] + 2))
    t2_sea_rate.append((sea2[0] + 1) / (sea2[1] + 2))
    
    # 7. Venue Toss Win Rate
    vt = venue_toss_stats.get(c, [0, 0])
    v_toss_winrate.append((vt[0] + 1) / (vt[1] + 2))
    
    # --- UPDATE HISTORICAL DATABASE AFTER READING MATCH PRIOR STATE ---
    # Update Elo
    e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
    e2 = 1.0 - e1
    s1 = 1.0 if w == t1 else 0.0
    s2 = 1.0 - s1
    elo_ratings[t1] = r1 + K * (s1 - e1)
    elo_ratings[t2] = r2 + K * (s2 - e2)
    
    # Update H2H
    h2h_stats[pair]['total'] += 1
    h2h_stats[pair][w] += 1
    
    # Update Form
    team_history[t1].append(1 if w == t1 else 0)
    team_history[t2].append(1 if w == t2 else 0)
    
    # Update Venue
    venue_stats[(c, t1)] = [st1[0] + (1 if w == t1 else 0), st1[1] + 1]
    venue_stats[(c, t2)] = [st2[0] + (1 if w == t2 else 0), st2[1] + 1]
    
    # Update Overall
    team_overall[t1][0] += 1 if w == t1 else 0
    team_overall[t1][1] += 1
    team_overall[t2][0] += 1 if w == t2 else 0
    team_overall[t2][1] += 1
    
    # Update Season
    season_team_stats[t1][0] += 1 if w == t1 else 0
    season_team_stats[t1][1] += 1
    season_team_stats[t2][0] += 1 if w == t2 else 0
    season_team_stats[t2][1] += 1
    
    # Update Venue Toss
    is_toss_win = (row['toss_winner'] == row['winner'])
    venue_toss_stats[c] = [vt[0] + (1 if is_toss_win else 0), vt[1] + 1]

# Attach created features
df['elo_team1'] = elo_t1
df['elo_team2'] = elo_t2
df['elo_diff'] = np.array(elo_t1) - np.array(elo_t2)
df['h2h_t1_winrate'] = h2h_rate
df['t1_winrate_l5'] = t1_w5
df['t2_winrate_l5'] = t2_w5
df['form_diff_l5'] = np.array(t1_w5) - np.array(t2_w5)
df['t1_winrate_l10'] = t1_w10
df['t2_winrate_l10'] = t2_w10
df['form_diff_l10'] = np.array(t1_w10) - np.array(t2_w10)
df['t1_venue_winrate'] = t1_v_rate
df['t2_venue_winrate'] = t2_v_rate
df['venue_winrate_diff'] = np.array(t1_v_rate) - np.array(t2_v_rate)
df['t1_overall_winrate'] = t1_ov_rate
df['t2_overall_winrate'] = t2_ov_rate
df['overall_winrate_diff'] = np.array(t1_ov_rate) - np.array(t2_ov_rate)
df['t1_season_winrate'] = t1_sea_rate
df['t2_season_winrate'] = t2_sea_rate
df['season_winrate_diff'] = np.array(t1_sea_rate) - np.array(t2_sea_rate)
df['venue_toss_winrate'] = v_toss_winrate

# Target and toss flags
df['target'] = (df['winner'] == df['team1']).astype(int)
df['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)

match_mapped = {
    "League": 0, "3rd Place Play-Off": 1, "Elimination Final": 2, "Eliminator": 2,
    "Qualifier 1": 3, "Qualifier 2": 4, "Semi Final": 5, "Final": 6
}
df['match_type_encoded'] = df['match_type'].map(match_mapped).fillna(0).astype(int)
df['toss_decision_encoded'] = df['toss_decision'].map({'field': 0, 'bat': 1}).fillna(0).astype(int)
df['t1_bat_first'] = ((df['is_toss_winner_team1'] == 1) & (df['toss_decision_encoded'] == 1)) | ((df['is_toss_winner_team1'] == 0) & (df['toss_decision_encoded'] == 0))
df['t1_bat_first'] = df['t1_bat_first'].astype(int)

df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

# Filter feature columns
features = [
    'season_year', 'elo_team1', 'elo_team2', 'elo_diff',
    'h2h_t1_winrate', 't1_winrate_l5', 't2_winrate_l5', 'form_diff_l5',
    't1_winrate_l10', 't2_winrate_l10', 'form_diff_l10',
    't1_venue_winrate', 't2_venue_winrate', 'venue_winrate_diff',
    't1_overall_winrate', 't2_overall_winrate', 'overall_winrate_diff',
    't1_season_winrate', 't2_season_winrate', 'season_winrate_diff',
    'venue_toss_winrate', 'is_toss_winner_team1', 'toss_decision_encoded',
    't1_bat_first', 'match_type_encoded', 'month', 'day_of_week', 'is_weekend'
]

X = df[features]
y = df['target']

# Train test split (80-20, maintaining shuffle with random_state=42 or time-based)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Candidate Models
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
xgb = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, random_state=42, eval_metric='logloss')
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=42)
et = ExtraTreesClassifier(n_estimators=300, max_depth=6, random_state=42)
hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.03, random_state=42)
lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('lr', LogisticRegression(C=0.05, max_iter=1000, random_state=42))],
    voting='soft'
)

stacking_clf = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('et', et)],
    final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=42)
)

models = {
    "Logistic Regression": (lr, True),
    "Random Forest": (rf, False),
    "XGBoost": (xgb, False),
    "Gradient Boosting": (gb, False),
    "Extra Trees": (et, False),
    "Hist Gradient Boosting": (hgb, False),
    "Voting Ensemble": (voting_clf, False),
    "Stacking Classifier": (stacking_clf, False)
}

results = {}
for name, (model, needs_scaling) in models.items():
    X_tr = x_train_scaled if needs_scaling else x_train
    X_te = x_test_scaled if needs_scaling else x_test
    
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    
    results[name] = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
        "F1-Score": round(f1_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "Log Loss": round(log_loss(y_test, probs), 4)
    }

res_df = pd.DataFrame(results).T.sort_values(by="Accuracy", ascending=False)
print("--- MODEL EVALUATION RESULTS ---")
print(res_df)
