import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_FILE = Path("matches.csv")
MODEL_FILE = Path("model_artifacts.pkl")


def load_and_prepare():
    df = pd.read_csv(DATA_FILE)

    team_mapping = {
        'Mumbai Indians': 'MI', 'Royal Challengers Bangalore': 'RCB',
        'Royal Challengers Bengaluru': 'RCB', 'Chennai Super Kings': 'CSK',
        'Kolkata Knight Riders': 'KKR', 'Delhi Daredevils': 'DC',
        'Delhi Capitals': 'DC', 'Kings XI Punjab': 'PBKS',
        'Punjab Kings': 'PBKS', 'Rajasthan Royals': 'RR',
        'Sunrisers Hyderabad': 'SRH', 'Deccan Chargers': 'HDC',
        'Gujarat Titans': 'GT', 'Lucknow Super Giants': 'LSG',
        'Rising Pune Supergiant': 'RPS', 'Rising Pune Supergiants': 'RPS',
        'Kochi Tuskers Kerala': 'KTK', 'Pune Warriors': 'PWI',
        'Gujarat Lions': 'GL'
    }

    venue_to_city_map = {
        'M Chinnaswamy Stadium': 'Bangalore', 'M.Chinnaswamy Stadium': 'Bangalore',
        'M Chinnaswamy Stadium, Bengaluru': 'Bangalore',
        'Punjab Cricket Association Stadium, Mohali': 'Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Mohali',
        'Punjab Cricket Association IS Bindra Stadium': 'Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Mohali',
        'Feroz Shah Kotla': 'Delhi', 'Arun Jaitley Stadium': 'Delhi',
        'Arun Jaitley Stadium, Delhi': 'Delhi', 'Wankhede Stadium': 'Mumbai',
        'Wankhede Stadium, Mumbai': 'Mumbai', 'Dr DY Patil Sports Academy': 'Mumbai',
        'Dr DY Patil Sports Academy, Mumbai': 'Mumbai', 'Brabourne Stadium': 'Mumbai',
        'Brabourne Stadium, Mumbai': 'Mumbai', 'Eden Gardens': 'Kolkata',
        'Eden Gardens, Kolkata': 'Kolkata', 'Sawai Mansingh Stadium': 'Jaipur',
        'Sawai Mansingh Stadium, Jaipur': 'Jaipur',
        'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',
        'Rajiv Gandhi International Stadium': 'Hyderabad',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Hyderabad',
        'MA Chidambaram Stadium, Chepauk': 'Chennai', 'MA Chidambaram Stadium': 'Chennai',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'Chennai', 'Newlands': 'Cape Town',
        "St George's Park": 'Gqeberha', 'Kingsmead': 'Durban',
        'SuperSport Park': 'Centurion', 'Buffalo Park': 'East London',
        'New Wanderers Stadium': 'Johannesburg', 'De Beers Diamond Oval': 'Kimberley',
        'OUTsurance Oval': 'Bloemfontein', 'Sardar Patel Stadium, Motera': 'Ahmedabad',
        'Narendra Modi Stadium, Ahmedabad': 'Ahmedabad', 'Barabati Stadium': 'Cuttack',
        'Vidarbha Cricket Association Stadium, Jamtha': 'Nagpur',
        'Himachal Pradesh Cricket Association Stadium': 'Dharamsala',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Dharamsala',
        'Nehru Stadium': 'Kochi', 'Holkar Cricket Stadium': 'Indore',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Visakhapatnam',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Visakhapatnam',
        'Subrata Roy Sahara Stadium': 'Pune', 'Maharashtra Cricket Association Stadium': 'Pune',
        'Maharashtra Cricket Association Stadium, Pune': 'Pune',
        'Shaheed Veer Narayan Singh International Stadium': 'Raipur',
        'JSCA International Stadium Complex': 'Ranchi', 'Sheikh Zayed Stadium': 'Abu Dhabi',
        'Zayed Cricket Stadium, Abu Dhabi': 'Abu Dhabi', 'Sharjah Cricket Stadium': 'Sharjah',
        'Dubai International Cricket Stadium': 'Dubai', 'Saurashtra Cricket Association Stadium': 'Rajkot',
        'Green Park': 'Kanpur',
        'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Lucknow',
        'Barsapara Cricket Stadium, Guwahati': 'Guwahati',
        'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur': 'Mullanpur'
    }

    match_type_map = {
        'League': 0, '3rd Place Play-Off': 1, 'Elimination Final': 2,
        'Eliminator': 2, 'Qualifier 1': 3, 'Qualifier 2': 4,
        'Semi Final': 5, 'Final': 6
    }

    team_cols = ['team1', 'team2', 'winner', 'toss_winner']
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].map(team_mapping).fillna(df[col])

    df.dropna(subset=['winner'], inplace=True)
    df['target'] = (df['winner'] == df['team1']).astype(int)

    df['city'] = df['city'].map(venue_to_city_map).fillna(df['city'])
    if 'venue' in df.columns:
        df['city'] = df['city'].fillna(df['venue'].map(venue_to_city_map))
    df['city'] = df['city'].fillna('Unknown')

    df['season_year'] = df['season'].astype(str).str.extract(r'(\d{4})').astype(int)
    df.drop(columns=['season'], inplace=True)

    df['match_type'] = df['match_type'].map(match_type_map).fillna(0).astype(int)

    toss_map = {'field': 0, 'bat': 1}
    df['toss_decision'] = df['toss_decision'].map(toss_map).fillna(0).astype(int)

    df['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)
    df['t1_bat_first'] = (
        ((df['is_toss_winner_team1'] == 1) & (df['toss_decision'] == 1)) |
        ((df['is_toss_winner_team1'] == 0) & (df['toss_decision'] == 0))
    ).astype(int)

    df = df.sort_values('season_year').reset_index(drop=True)

    team_stats = {}
    for team in set(df['team1']).union(set(df['team2'])):
        team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
        team_wins = team_matches[team_matches['winner'] == team].shape[0]
        team_stats[team] = team_wins / max(team_matches.shape[0], 1)

    df['team1_win_rate'] = df['team1'].map(team_stats).fillna(0.5)
    df['team2_win_rate'] = df['team2'].map(team_stats).fillna(0.5)
    df['win_rate_diff'] = df['team1_win_rate'] - df['team2_win_rate']

    h2h = {}
    for _, row in df.iterrows():
        t1, t2, w = row['team1'], row['team2'], row['winner']
        key = tuple(sorted([t1, t2]))
        if key not in h2h:
            h2h[key] = {'t1': t1, 't1_wins': 0, 'total': 0}
        h2h[key]['total'] += 1
        if w == t1:
            h2h[key]['t1_wins'] += 1
    h2h_rates = {k: v['t1_wins'] / max(v['total'], 1) for k, v in h2h.items()}
    df['h2h_team1_win_rate'] = df.apply(
        lambda r: h2h_rates.get(tuple(sorted([r['team1'], r['team2']])), 0.5), axis=1
    )

    city_stats = {}
    for city in df['city'].unique():
        city_matches = df[df['city'] == city]
        if city_matches.shape[0] == 0:
            city_stats[city] = 0.5
            continue
        city_wins = city_matches[city_matches['winner'] == city_matches['team1']].shape[0]
        city_stats[city] = city_wins / city_matches.shape[0]
    df['city_team1_win_rate'] = df['city'].map(city_stats).fillna(0.5)

    feature_cols = [
        'team1', 'team2', 'toss_winner', 'city', 'match_type',
        'toss_decision', 'is_toss_winner_team1', 't1_bat_first',
        'team1_win_rate', 'team2_win_rate', 'win_rate_diff',
        'h2h_team1_win_rate', 'city_team1_win_rate'
    ]

    X = df[feature_cols].copy()
    y = df['target'].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    te_city = TargetEncoder(smooth='auto', cv=5)
    te_t1 = TargetEncoder(smooth='auto', cv=5)
    te_t2 = TargetEncoder(smooth='auto', cv=5)
    te_tw = TargetEncoder(smooth='auto', cv=5)

    x_train['city_enc'] = te_city.fit_transform(x_train[['city']], y_train)
    x_test['city_enc'] = te_city.transform(x_test[['city']])

    x_train['team1_enc'] = te_t1.fit_transform(x_train[['team1']], y_train)
    x_test['team1_enc'] = te_t1.transform(x_test[['team1']])

    x_train['team2_enc'] = te_t2.fit_transform(x_train[['team2']], y_train)
    x_test['team2_enc'] = te_t2.transform(x_test[['team2']])

    x_train['toss_winner_enc'] = te_tw.fit_transform(x_train[['toss_winner']], y_train)
    x_test['toss_winner_enc'] = te_tw.transform(x_test[['toss_winner']])

    drop_cols = ['city', 'team1', 'team2', 'toss_winner']
    x_train = x_train.drop(columns=drop_cols)
    x_test = x_test.drop(columns=drop_cols)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(C=0.1, max_iter=3000, solver='liblinear', random_state=42)
    model.fit(x_train_scaled, y_train)

    preds = model.predict(x_test_scaled)
    probs = model.predict_proba(x_test_scaled)[:, 1]
    print(f"Test Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Test ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")

    # Collect all options for UI
    all_teams = sorted(set(df['team1'].unique().tolist() + df['team2'].unique().tolist()))
    all_cities = sorted(df['city'].unique().tolist())
    all_match_types = ['League', '3rd Place Play-Off', 'Elimination Final', 'Eliminator', 'Qualifier 1', 'Qualifier 2', 'Semi Final', 'Final']

    artifacts = {
        'model': model,
        'scaler': scaler,
        'te_city': te_city,
        'te_t1': te_t1,
        'te_t2': te_t2,
        'te_tw': te_tw,
        'team_stats': team_stats,
        'h2h_rates': h2h_rates,
        'city_stats': city_stats,
        'venue_to_city_map': venue_to_city_map,
        'match_type_map': match_type_map,
        'toss_map': toss_map,
        'all_teams': all_teams,
        'all_cities': all_cities,
        'all_match_types': all_match_types,
        'feature_cols': feature_cols,
        'drop_cols': drop_cols,
    }

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"Artifacts saved to {MODEL_FILE}")


if __name__ == '__main__':
    load_and_prepare()
