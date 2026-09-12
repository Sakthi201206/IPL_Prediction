import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODEL_FILE = Path("model_artifacts.pkl")

TEAM_COLORS = {
    'MI': '#004BA0',
    'CSK': "#C0C700",
    'KKR': '#3A225D',
    'RCB': '#D4213D',
    'DC': '#0078C7',
    'PBKS': '#DD1F2D',
    'RR': '#EA1A85',
    'SRH': '#FF822A',
    'GT': '#1B2133',
    'LSG': "#00A07D",
    'HDC': "#0014C7",
    'RPS': "#C810C8",
    'KTK': '#E61F26',
    'PWI': '#E61F26',
    'GL': '#DD1F2D',
}


@st.cache_resource
def load_artifacts():
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)


def predict(artifacts, user_input: dict):
    model = artifacts['model']
    scaler = artifacts['scaler']
    te_city = artifacts['te_city']
    te_t1 = artifacts['te_t1']
    te_t2 = artifacts['te_t2']
    te_tw = artifacts['te_tw']
    team_stats = artifacts['team_stats']
    h2h_rates = artifacts['h2h_rates']
    city_stats = artifacts['city_stats']
    match_type_map = artifacts['match_type_map']
    toss_map = artifacts['toss_map']
    drop_cols = artifacts['drop_cols']

    df = pd.DataFrame([user_input])

    df['match_type'] = df['match_type'].map(match_type_map).fillna(0).astype(int)
    df['toss_decision'] = df['toss_decision'].map(toss_map).fillna(0).astype(int)
    df['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)
    df['t1_bat_first'] = (
        ((df['is_toss_winner_team1'] == 1) & (df['toss_decision'] == 1)) |
        ((df['is_toss_winner_team1'] == 0) & (df['toss_decision'] == 0))
    ).astype(int)

    df['team1_win_rate'] = df['team1'].map(team_stats).fillna(0.5)
    df['team2_win_rate'] = df['team2'].map(team_stats).fillna(0.5)
    df['win_rate_diff'] = df['team1_win_rate'] - df['team2_win_rate']

    h2h_key = tuple(sorted([df['team1'].iloc[0], df['team2'].iloc[0]]))
    df['h2h_team1_win_rate'] = h2h_rates.get(h2h_key, 0.5)

    df['city_team1_win_rate'] = df['city'].map(city_stats).fillna(0.5)

    df['city_enc'] = te_city.transform(df[['city']])
    df['team1_enc'] = te_t1.transform(df[['team1']])
    df['team2_enc'] = te_t2.transform(df[['team2']])
    df['toss_winner_enc'] = te_tw.transform(df[['toss_winner']])

    df = df.drop(columns=drop_cols)

    final_cols = [
        'match_type', 'toss_decision', 'is_toss_winner_team1', 't1_bat_first',
        'team1_win_rate', 'team2_win_rate', 'win_rate_diff',
        'h2h_team1_win_rate', 'city_team1_win_rate',
        'city_enc', 'team1_enc', 'team2_enc', 'toss_winner_enc'
    ]
    X = df[final_cols].copy()

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0, 1]

    return pred, prob


def draw_prediction_bar(team1, team2, prob):
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    color1 = TEAM_COLORS.get(team1, '#555555')
    color2 = TEAM_COLORS.get(team2, '#555555')

    ax.barh(0.5, prob, height=0.6, left=0, color=color1, edgecolor='white', linewidth=1.5)
    ax.barh(0.5, 1 - prob, height=0.6, left=prob, color=color2, edgecolor='white', linewidth=1.5)

    ax.text(prob / 2, 0.5, f"{team1}\n{prob:.1%}", ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.text(prob + (1 - prob) / 2, 0.5, f"{team2}\n{(1 - prob):.1%}", ha='center', va='center', color='white', fontsize=11, fontweight='bold')

    ax.axis('off')
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏")
    st.title("🏏 IPL Match Predictor")
    st.write("Predict whether **team1** will win the match based on match metadata.")

    artifacts = load_artifacts()

    all_teams = artifacts['all_teams']
    all_cities = artifacts['all_cities']
    all_match_types = artifacts['all_match_types']

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1", all_teams, index=0)
            toss_winner = st.selectbox("Toss Winner", all_teams, index=0)
            city = st.selectbox("City", all_cities, index=0)
        with col2:
            team2 = st.selectbox("Team 2", all_teams, index=1)
            match_type = st.selectbox("Match Type", all_match_types, index=0)
            toss_decision = st.selectbox("Toss Decision", ["bat", "field"], index=0)

        submitted = st.form_submit_button("Predict Winner")

    if submitted:
        if team1 == team2:
            st.warning("Team 1 and Team 2 should be different.")
        else:
            user_input = {
                'team1': team1,
                'team2': team2,
                'toss_winner': toss_winner,
                'city': city,
                'match_type': match_type,
                'toss_decision': toss_decision,
            }

            pred, prob = predict(artifacts, user_input)
            winner = team1 if pred == 1 else team2

            st.subheader("Prediction")
            team1_prob = prob
            team2_prob = 1 - prob
            predicted_prob = team1_prob if pred == 1 else team2_prob
            st.metric(label="Predicted Winner", value=winner)
            st.metric(label="Predicted Winner Win Probability", value=f"{predicted_prob:.2%}")

            fig = draw_prediction_bar(team1, team2, prob)
            st.pyplot(fig)

            if pred == 1:
                st.success(f"{team1} is predicted to win!")
            else:
                st.info(f"{team2} is predicted to win!")


if __name__ == '__main__':
    main()
