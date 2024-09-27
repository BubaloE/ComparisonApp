import pyodbc
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Konfiguration af siden
st.set_page_config(page_title="Comparison Report", page_icon="📊")

# Define connection parameters
server = '52.166.191.42,4022'
database = 'DSA'
username = 'sje'
password = 'Haderslev2024'
driver = '{ODBC Driver 17 for SQL Server}'

# Establish the connection
conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')

# Function to execute queries and return a DataFrame
@st.cache_data
def ejecutar_consulta(query):
    return pd.read_sql(query, conn)

# Hent de fire datasæt ved at udføre separate SQL-forespørgsler
df_player_stats_total = ejecutar_consulta("SELECT * FROM [DSA].[WyScout].[PlayerAdvanceStats_Total] WHERE SeasonId = 189918")
df_player_stats_avg = ejecutar_consulta("SELECT * FROM [DSA].[WyScout].[PlayerAdvanceStats_Average] WHERE SeasonId = 189918")
df_player_stats_percent = ejecutar_consulta("SELECT * FROM [DSA].[WyScout].[PlayerAdvanceStats_Percent] WHERE SeasonId = 189918")
df_players = ejecutar_consulta("SELECT * FROM [DSA].[WyScout].[Players] WHERE SeasonId = '189918'")
df_teams = ejecutar_consulta("SELECT * FROM [DSA].[WyScout].[Teams]")

# Displaying a sample of the data
st.write(df_player_stats_total.head())

# Filtrer hold til kun at inkludere dem med SeasonId = 189918
df_teams_filtered = df_teams[df_teams['TeamId'].isin(df_players['CurrentTeamId'].unique())]

# Opret FullName ved at kombinere FirstName og LastName i df_players
df_players['FullName'] = df_players['FirstName'] + ' ' + df_players['LastName']

# Tilføj præfikser til kolonnenavnene i hvert datasæt
df_player_stats_avg = df_player_stats_avg.add_prefix('Avg_')
df_player_stats_total = df_player_stats_total.add_prefix('Total_')
df_player_stats_percent = df_player_stats_percent.add_prefix('Pct_')

# Bevar PlayerId i alle datasæt for at kunne merge dem
df_player_stats_avg = df_player_stats_avg.rename(columns={'Avg_PlayerId': 'PlayerId'})
df_player_stats_total = df_player_stats_total.rename(columns={'Total_PlayerId': 'PlayerId'})
df_player_stats_percent = df_player_stats_percent.rename(columns={'Pct_PlayerId': 'PlayerId'})

# Merge spillere og hold
df_players = df_players.merge(df_teams_filtered[['TeamId', 'OfficialName']], left_on='CurrentTeamId', right_on='TeamId', how='left')

# Merge spillere med deres holdnavne til de tre statistiksæt
df_player_stats_total = df_player_stats_total.merge(df_players[['PlayerWyId', 'FullName', 'RoleName', 'OfficialName']], left_on='PlayerId', right_on='PlayerWyId', how='left')
df_player_stats_avg = df_player_stats_avg.merge(df_players[['PlayerWyId', 'FullName', 'RoleName', 'OfficialName']], left_on='PlayerId', right_on='PlayerWyId', how='left')
df_player_stats_percent = df_player_stats_percent.merge(df_players[['PlayerWyId', 'FullName', 'RoleName', 'OfficialName']], left_on='PlayerId', right_on='PlayerWyId', how='left')

# Vælg de ønskede variabler fra Percent datasættet og omdøb dem
df_player_stats_percent_cleaned = df_player_stats_percent[[
    'OfficialName',
    'FullName',
    'Pct_duelsWon', 
    'Pct_defensiveDuelsWon', 
    'Pct_offensiveDuelsWon', 
    'Pct_aerialDuelsWon', 
    'Pct_successfulPasses', 
    'Pct_successfulPassesToFinalThird',
    'Pct_successfulCrosses', 
    'Pct_successfulDribbles', 
    'Pct_shotsOnTarget', 
    'Pct_goalConversion', 
    'Pct_successfulForwardPasses', 
    'Pct_successfulBackPasses', 
    'Pct_successfulThroughPasses', 
    'Pct_successfulVerticalPasses', 
    'Pct_successfulLongPasses', 
    'Pct_successfulShotAssists', 
    'Pct_successfulProgressivePasses', 
    'Pct_dribblesAgainstWon', 
    'Pct_fieldAerialDuelsWon', 
    'Pct_gkSaves', 
    'Pct_gkAerialDuelsWon', 
    'Pct_newDuelsWon', 
    'Pct_newDefensiveDuelsWon', 
    'Pct_newOffensiveDuelsWon', 
    'Pct_newSuccessfulDribbles', 
    'Pct_successfulLateralPasses'
]].rename(columns={
    'Pct_duelsWon': 'Duels Won (%)',
    'Pct_defensiveDuelsWon': 'Defensive Duels Won (%)',
    'Pct_offensiveDuelsWon': 'Offensive Duels Won (%)',
    'Pct_aerialDuelsWon': 'Aerial Duels Won (%)',
    'Pct_successfulPasses': 'Passes (%)',
    'Pct_successfulPassesToFinalThird': 'Passes to Final 3rd (%)',
    'Pct_successfulCrosses': 'Crosses (%)',
    'Pct_successfulDribbles': 'Dribbles (%)',
    'Pct_shotsOnTarget': 'Shots (On Target %)',
    'Pct_goalConversion': 'Shots (Conversion %)',
    'Pct_successfulForwardPasses': 'Forward Passes (%)',
    'Pct_successfulBackPasses': 'Back Passes (%)',
    'Pct_successfulThroughPasses': 'Through Passes (%)',
    'Pct_successfulVerticalPasses': 'Vertical Passes (%)',
    'Pct_successfulLongPasses': 'Long Passes (%)',
    'Pct_successfulShotAssists': 'Shot Assists (%)',
    'Pct_successfulProgressivePasses': 'Progressive Passes (%)',
    'Pct_dribblesAgainstWon': 'Dribbles Against (%)',
    'Pct_fieldAerialDuelsWon': 'Field Aerial Duels (%)',
    'Pct_gkSaves': 'GK Saves (%)',
    'Pct_gkAerialDuelsWon': 'GK Aerial Duels (%)',
    'Pct_newDuelsWon': 'New Duels (%)',
    'Pct_newDefensiveDuelsWon': 'New Defensive Duels (%)',
    'Pct_newOffensiveDuelsWon': 'New Offensive Duels (%)',
    'Pct_newSuccessfulDribbles': 'New Dribbles (%)',
    'Pct_successfulLateralPasses': 'Lateral Passes (%)'
})

# Funktion til at normalisere værdier med Min-Max Normalisering
def normalize_min_max(df, selected_vars):
    normalized_df = df.copy()
    for var in selected_vars:
        min_value = df[var].min()
        max_value = df[var].max()
        normalized_df[var] = (df[var] - min_value) / (max_value - min_value)
    return normalized_df

# Streamlit UI - Titel
st.markdown(
    "<h1 style='text-align: center; font-size: 40px; color: black;'>COMPARISON REPORT</h1>", 
    unsafe_allow_html=True
)

# Valg af datasæt
dataset_option = st.selectbox(
    'Choose dataformat:',
    ('Total', 'Average', 'Percent')
)

# Valg af hold for hver spiller
team_option_1 = st.selectbox('Team for Player 1', df_teams_filtered['OfficialName'].unique())

# Filtrer spillere baseret på det valgte hold for hver spiller
df_players_filtered_1 = df_players[df_players['OfficialName'] == team_option_1]

# Spillervalg baseret på FullName fra det filtrerede datasæt
player_1 = st.selectbox('Choose Player 1', df_players_filtered_1['FullName'].unique())

# Valg af hold for anden spiller
team_option_2 = st.selectbox('Team for Player 2', df_teams_filtered['OfficialName'].unique())

df_players_filtered_2 = df_players[df_players['OfficialName'] == team_option_2]

# Spillervalg baseret på FullName fra det filtrerede datasæt
player_2 = st.selectbox('Choose Player 2', df_players_filtered_2['FullName'].unique())

# Filtrer spillere baseret på det valgte hold for hver spiller
df_players_filtered_1 = df_players[df_players['OfficialName'] == team_option_1]
df_players_filtered_2 = df_players[df_players['OfficialName'] == team_option_2]

# Baseret på valg af datasæt, vælger vi de relevante variabler
if dataset_option == 'Total':
    df_selected_1 = df_player_stats_total[df_player_stats_total['OfficialName'] == team_option_1]
    df_selected_2 = df_player_stats_total[df_player_stats_total['OfficialName'] == team_option_2]
elif dataset_option == 'Average':
    df_selected_1 = df_player_stats_avg[df_player_stats_avg['OfficialName'] == team_option_1]
    df_selected_2 = df_player_stats_avg[df_player_stats_avg['OfficialName'] == team_option_2]
else:
    df_selected_1 = df_player_stats_percent_cleaned[df_player_stats_percent_cleaned['OfficialName'] == team_option_1]
    df_selected_2 = df_player_stats_percent_cleaned[df_player_stats_percent_cleaned['OfficialName'] == team_option_2]

# Tilføj multiselect for variabler baseret på det valgte datasæt
# Vi udelukker 'PlayerId', 'FullName', og 'RoleName' fra valgmulighederne
available_vars = df_selected_1.columns.tolist()[4:-2]  # Justeret for at undgå at inkludere irrelevante kolonner som IDs

selected_vars = st.multiselect('Choose variables / parameters', available_vars, default=available_vars[:5])

# Bevar en kopi af de oprindelige datasæt for visning i tabellen
df_selected_1_original = df_selected_1.copy()
df_selected_2_original = df_selected_2.copy()

# Normaliser data for Total og Average datasæt (kun til radar plot)
if dataset_option == 'Total' or dataset_option == 'Average':
    df_selected_1 = normalize_min_max(df_selected_1, selected_vars)
    df_selected_2 = normalize_min_max(df_selected_2, selected_vars)

# Filtrer data for de valgte spillere og rund værdierne til 1 decimal for det normaliserede datasæt (til radar plot)
player_1_data = df_selected_1[df_selected_1['FullName'] == player_1][selected_vars].iloc[0].round(1)
player_2_data = df_selected_2[df_selected_2['FullName'] == player_2][selected_vars].iloc[0].round(1)

# Filtrer data for de valgte spillere fra det oprindelige datasæt til visning i tabellen
player_1_data_original = df_selected_1_original[df_selected_1_original['FullName'] == player_1][selected_vars].iloc[0].round(1)
player_2_data_original = df_selected_2_original[df_selected_2_original['FullName'] == player_2][selected_vars].iloc[0].round(1)

# Opret en tabel til at vise de oprindelige værdier mellem de to spillere
comparison_df_original = pd.DataFrame({
    'Variabel': selected_vars,
    player_1: player_1_data_original.values,
    player_2: player_2_data_original.values
})

# Tilføj en ny kolonne for at vise vinderen og forskellen
comparison_df_original['Difference'] = comparison_df_original.apply(
    lambda row: f"{player_1} +{row[player_1] - row[player_2]:.1f}" if row[player_1] > row[player_2] 
    else f"{player_2} +{row[player_2] - row[player_1]:.1f}", axis=1)

# Formatér værdierne med én decimal i tabellen
comparison_df_original[player_1] = comparison_df_original[player_1].map('{:.1f}'.format)
comparison_df_original[player_2] = comparison_df_original[player_2].map('{:.1f}'.format)

# Funktion til at plotte radar chart med mindre tekst
def plot_radar(player_1_data, player_2_data, labels, player_1_name, player_2_name):
    from math import pi
    import matplotlib.pyplot as plt

    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Luk cirklen

    # Spillernes værdier og slut værdien for at lukke cirklen
    player_1_values = player_1_data.tolist()
    player_1_values += player_1_values[:1]

    player_2_values = player_2_data.tolist()
    player_2_values += player_2_values[:1]

    # Justér størrelsen på plottet og teksten med figsize og font size
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))  # Ændret figsize til mindre (4, 4)

    # Plot for spiller 1 (Blå farve)
    ax.plot(angles, player_1_values, linewidth=1, linestyle='solid', label=player_1_name, color='blue')
    ax.fill(angles, player_1_values, 'blue', alpha=0.1)

    # Plot for spiller 2 (Rød farve)
    ax.plot(angles, player_2_values, linewidth=1, linestyle='solid', label=player_2_name, color='red')
    ax.fill(angles, player_2_values, 'red', alpha=0.1)

    # Fjern numeriske y-aksel labels
    ax.yaxis.set_visible(False)

    # Indstil variabelnavne rundt om plottet med mindre font size
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=6)  # Mindre tekststørrelse for variabelnavne

    # Roter labels for at følge cirklen med mindre font size
    for label, angle in zip(ax.get_xticklabels(), angles):
        angle_deg = angle * 180 / pi  # Konverter vinkel til grader
        if angle_deg <= 90 or angle_deg > 270:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        label.set_rotation(angle_deg - 90)  # Roter labelen, så den følger radaren

    # Juster radials
    ax.set_rlabel_position(0)  # Start vinkel for radials

    # Tilføj mindre legende over plottet
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=6)  # Mindre legendetekst

    st.pyplot(fig)

# Funktion til at beregne en farve baseret på forskellen
def color_gradient(value, max_diff):
    cmap = plt.get_cmap('Oranges')  # Brug 'Greens' colormap fra matplotlib
    norm = mcolors.Normalize(vmin=0, vmax=max_diff)
    return mcolors.to_hex(cmap(norm(value)))

# Funktion til at fremhæve celler med dynamisk farveskala
def highlight_better(row):
    player_1_val = float(row[player_1])
    player_2_val = float(row[player_2])
    
    diff = abs(player_1_val - player_2_val)
    
    max_diff = max(player_1_val, player_2_val) * 0.5  # For at give en relativ skala for farver
    
    color_1 = color_gradient(diff if player_1_val > player_2_val else 0, max_diff)
    color_2 = color_gradient(diff if player_2_val > player_1_val else 0, max_diff)
    
    color_1 = 'background-color: {}'.format(color_1) if player_1_val > player_2_val else 'background-color: white'
    color_2 = 'background-color: {}'.format(color_2) if player_2_val > player_1_val else 'background-color: white'
    
    return [color_1, color_2]  # Returnér kun to værdier, en for hver spillerkolonne

# Anvend betinget formatering på DataFrame, kun for spillerkolonnerne
styled_comparison_df_original = comparison_df_original.style.apply(highlight_better, subset=[player_1, player_2], axis=1)

plot_radar(player_1_data, player_2_data, selected_vars, player_1, player_2)

# Vis sammenligningstabel efter radarplottet
st.markdown(
    "<p style='text-align: center; font-size: 24px; color: black;'>Comparing Players</p>", 
    unsafe_allow_html=True
)
st.dataframe(styled_comparison_df_original)
