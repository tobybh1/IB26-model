# Cleaning the data and encoding features

import pandas as pd
import numpy as np

# ===== load data =====
runner_df = pd.read_excel('IB database.xlsx', sheet_name='Runners',
                          usecols=['name', 'college', 'division', 'year', 
                                   'placing'])
team_df = pd.read_excel('IB database.xlsx', sheet_name='Teams',
                        usecols=['college', 'division', 'year', 'score'])
college_df = pd.read_excel('IB database.xlsx', sheet_name='Colleges')
rogaine_df = pd.read_excel('night_rogaine_results22-26.xlsx')

# ===== team-level features, =====
# ===== e.g. how a specific team has performed in recent  years =====
team_df = team_df.sort_values(['college', 'division', 'year'])

team_df['last_team_score'] = (
    team_df
    .groupby(['college', 'division'])['score']
    .shift()
    )

team_df['avg_team_score'] = (
    team_df
    .groupby(['college', 'division'])['score']
    .transform(lambda x: x.shift().expanding().mean())
    )

team_df['best_team_score'] = (
    team_df
    .groupby(['college', 'division'])['score']
    .transform(lambda x: x.shift().expanding().max())
    )

team_df['worst_team_score'] = (
    team_df
    .groupby(['college', 'division'])['score']
    .transform(lambda x: x.shift().expanding().min())
    )

# ===== college-level features =====
college_df = college_df.sort_values(['college', 'year'])

college_df['college_last_placing'] = (
    college_df
    .groupby('college')['placing']
    .shift()
    )

college_df['college_avg_placing'] = (
    college_df
    .groupby('college')['placing']
    .transform(lambda x: x.shift().expanding().mean())
    )

college_df['college_best_placing'] = (
    college_df
    .groupby('college')['placing']
    .transform(lambda x: x.shift().expanding().min())
    )

college_df['college_worst_placing'] = (
    college_df
    .groupby('college')['placing']
    .transform(lambda x: x.shift().expanding().max())
    )

# ===== runner's IB history =====
runner_hist = runner_df.merge(
    team_df[['college','division','year','score']],
    on=['college','division','year'],
    how='left'
)

runner_hist = runner_hist.sort_values(['name', 'year'])

runner_hist['yoe'] = (
    runner_hist
    .groupby('name')
    .cumcount()
    )

runner_hist['last_div'] = (
    runner_hist
    .groupby('name')['division']
    .shift()
    )

runner_hist['avg_div'] = (
    runner_hist
    .groupby('name')['division']
    .transform(lambda x: x.shift().expanding().mean())
    )

runner_hist['highest_div'] = (
    runner_hist
    .groupby('name')['division']
    .transform(lambda x: x.shift().expanding().min())
    )

runner_hist['lowest_div'] = (
    runner_hist
    .groupby('name')['division']
    .transform(lambda x: x.shift().expanding().max())
    )

runner_hist['last_score'] = (
    runner_hist
    .groupby('name')['score']
    .shift()
    )

runner_hist['avg_score'] = (
    runner_hist
    .groupby('name')['score']
    .transform(lambda x: x.shift().expanding().mean())
    )

runner_hist['best_score'] = (
    runner_hist
    .groupby('name')['score']
    .transform(lambda x: x.shift().expanding().max())
    )

runner_hist['worst_score'] = (
    runner_hist
    .groupby('name')['score']
    .transform(lambda x: x.shift().expanding().min())
    )

runner_hist['new_runner'] = (runner_hist['yoe'] == 0).astype(int)

runner_hist['dnf_dq'] = runner_hist['placing'].isin(['DNF', 'DQ']).astype(int)

# ===== runner's day/night rogaine history =====
rog_hist = rogaine_df[rogaine_df['name'].isin(runner_df['name'])]
rog_hist = rog_hist.sort_values(['name', 'year'])

rog_hist['rog_exp'] = (
    rog_hist
    .groupby('name')
    .cumcount() + 1
    )

rog_hist['last_rog'] = (
    rog_hist
    .groupby(['name', 'year'])['z-score']
    .transform('max')
    )

rog_hist['avg_rog'] = (
    rog_hist
    .groupby('name')['z-score']
    .transform(lambda x: x.expanding().mean())
    )

rog_hist['best_rog'] = (
    rog_hist
    .groupby('name')['z-score']
    .transform(lambda x: x.expanding().max())
    )

rog_hist['worst_rog'] = (
    rog_hist
    .groupby('name')['z-score']
    .transform(lambda x: x.expanding().min())
    )

rog_hist = (
    rog_hist
    .sort_values('rog_exp', ascending=False)
    .drop_duplicates(subset=['name', 'year'], keep='first')
)

# merging with runner's history
runner_hist = runner_hist.merge(
    rog_hist[['name','year','rog_exp', 'last_rog', 'avg_rog', 'best_rog', 
              'worst_rog']],
    on=['name','year'],
    how='left'
)
cols = ['rog_exp', 'last_rog', 'avg_rog', 'best_rog', 'worst_rog']

runner_hist[cols] = runner_hist.groupby('name')[cols].ffill()

# assign runners to their teams
runner_hist = runner_hist.merge(
    team_df[
        [
            'college',
            'division',
            'year',
            'avg_team_score',
            'last_team_score',
            'best_team_score',
            'worst_team_score'
            ]
        ],
    on=['college', 'division', 'year'],
    how='left'
    )

# assign runners to their colleges
runner_hist = runner_hist.merge(
    college_df[
        [
            'college',
            'year',
            'college_avg_placing',
            'college_last_placing',
            'college_best_placing',
            'college_worst_placing'
            ]
        ],
    on=['college', 'year'],
    how='left'
    )

runner_hist.to_csv('runner_hist.csv', index=False)

# ===== building the team features table =====
team_features = (
    runner_hist
    .groupby(['college', 'division', 'year'])
    .agg(
        # experience
        total_yoe=('yoe', 'sum'),
        max_yoe=('yoe', 'max'),
        
        # new runners
        num_new_runners=('new_runner', 'sum'),
        
        # division history
        avg_div=('avg_div', 'mean'),
        avg_last_div=('last_div', 'mean'),
        avg_highest_div=('highest_div', 'mean'),
        avg_lowest_div=('lowest_div', 'mean'),
        best_runner_div=('highest_div', 'min'),
        worst_runner_div=('lowest_div', 'max'),
        
        # score history
        avg_score=('avg_score', 'mean'),
        avg_last_score=('last_score', 'mean'),
        avg_best_score=('best_score', 'mean'),
        avg_worst_score=('worst_score', 'mean'),
        best_runner_score=('best_score', 'max'),
        worst_runner_score=('worst_score', 'min'),
        
        # historical team performance
        avg_team_score=('avg_team_score', 'first'),
        last_team_score=('last_team_score', 'first'),
        best_team_score=('best_team_score', 'first'),
        worst_team_score=('worst_team_score', 'first'),
        
        # historical college performance
        college_last_placing=('college_last_placing', 'first'),
        college_avg_placing=('college_avg_placing', 'first'),
        college_best_placing=('college_best_placing', 'first'),
        college_worst_placing=('college_worst_placing', 'first'),
        
        # DNFs/DQs
        sum_dnf_dq=('dnf_dq', 'sum'),
        
        # rogaine history
        # experience
        total_rogs=('rog_exp', 'sum'),
        max_rogs=('rog_exp', 'max'),
        
        # score history
        avg_rog=('avg_rog', 'mean'),
        avg_last_rog=('last_rog', 'mean'),
        avg_best_rog=('best_rog', 'mean'),
        avg_worst_rog=('worst_rog', 'mean'),
        best_rog=('best_rog', 'max'),
        worst_rog=('worst_rog', 'min')
        )
    .reset_index()
    )

# ===== add target variable =====
team_features = team_features.merge(
    team_df[['college', 'division', 'year', 'score']],
    on=['college', 'division', 'year'],
    how='left'
)

# Div X teams were set to Div X1, X2 etc. in Excel to ensure accurate score 
# calculations if there were multiple div x teams in the same division, so now 
# we convert them back to Div X
team_features['college'] = team_features['college'].str.replace(
    r'^Div X\d+', 'Div X', regex=True
)

# setting prev div x results to nan, e.g. div x div 2 2023 is not correlated 
# with div x div 2 2024
# not sure if there's a better way to deal with div x
team_features.loc[
    team_features['college'] == 'Div X',
    ['avg_team_score', 'last_team_score', 'best_team_score']
] = np.nan

team_features.to_csv('team_features.csv', index=False)

# correlation matrix for numeric columns only
corr_matrix = team_features.select_dtypes(include='number').corr()
corr_matrix.to_csv('correlation_matrix.csv')
