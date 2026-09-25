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

# ===== runner history =====
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

runner_hist['new_runner'] = (runner_hist['yoe'] == 0).astype(int)

runner_hist['dnf_dq'] = runner_hist['placing'].isin(['DNF', 'DQ']).astype(int)

# Assign runners to their teams
runner_hist = runner_hist.merge(
    team_df[
        [
            'college',
            'division',
            'year',
            'avg_team_score',
            'last_team_score',
            'best_team_score'
            ]
        ],
    on=['college', 'division', 'year'],
    how='left'
    )

# Assign runners to their colleges
runner_hist = runner_hist.merge(
    college_df[
        [
            'college',
            'year',
            'college_avg_placing',
            'college_last_placing',
            'college_best_placing'
            ]
        ],
    on=['college', 'year'],
    how='left'
    )

runner_hist.to_csv('runner_hist.csv', index=False)

# ===== building the team features table
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
        best_runner_div=('highest_div', 'min'),
        
        # score history
        avg_score=('avg_score', 'mean'),
        avg_last_score=('last_score', 'mean'),
        avg_best_score=('best_score', 'mean'),
        best_runner_score=('avg_score', 'max'),
        best_last_score=('last_score', 'max'),
        
        # historical team performance
        avg_team_score=('avg_team_score', 'first'),
        last_team_score=('last_team_score', 'first'),
        best_team_score=('best_team_score', 'first'),
        
        # historical college performance
        college_last_placing=('college_last_placing', 'first'),
        college_avg_placing=('college_avg_placing', 'first'),
        college_best_placing=('college_best_placing', 'first'),
        
        # DNFs/DQs
        sum_dnf_dq=('dnf_dq', 'sum')
        )
    .reset_index()
    )

# ===== Add target variable =====

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

# Setting prev div x results to nan
# Not sure if there's a better way to deal with div x
team_features.loc[
    team_features['college'] == 'Div X',
    ['avg_team_score', 'last_team_score', 'best_team_score']
] = np.nan

team_features.to_csv('team_features.csv', index=False)

# Correlation matrix for numeric columns only
corr_matrix = team_features.select_dtypes(include='number').corr()
corr_matrix.to_csv('correlation_matrix.csv')
