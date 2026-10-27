import json

with open("notebooks/02_elo.ipynb", "r") as f:
    nb = json.load(f)

# Cell 1 (index 1): Change imports
nb["cells"][1]["source"] = [
    "import sys\n",
    "sys.path.append('../src')\n",
    "from elo import EloEngine"
]

# Cell 2 (index 2): Change the example
nb["cells"][2]["source"] = [
    "engine = EloEngine()\n",
    "engine._ratings['TeamA'] = 1600\n",
    "engine._ratings['TeamB'] = 1500\n",
    "\n",
    "r_home, r_away, new_home, new_away = engine.update('TeamA', 'TeamB', home_goals=3, away_goals=0)\n",
    "\n",
    "print(f\"Old Elo — Home: {r_home:.1f}, Away: {r_away:.1f}\")\n",
    "print(f\"New Elo (3-0 win) — Home: {new_home:.1f}, Away: {new_away:.1f}\")"
]

# Cell 3 (index 3): Initialize ratings dict is no longer needed since engine does it, but we can just clear it.
nb["cells"][3]["source"] = [
    "# The EloEngine handles initializing default ratings automatically\n",
    "engine = EloEngine()"
]

# Cell 4 (index 4): Update iteration
nb["cells"][4]["source"] = [
    "home_elo_before = []\n",
    "away_elo_before = []\n",
    "\n",
    "for idx, row in df_clean.iterrows():\n",
    "    home_team = row['HomeTeam']\n",
    "    away_team = row['AwayTeam']\n",
    "    home_goals = int(row['FTHG'])\n",
    "    away_goals = int(row['FTAG'])\n",
    "    \n",
    "    # Engine update automatically handles the Margin of Victory (goals) \n",
    "    # and Home Advantage, and updates the internal ratings dictionary.\n",
    "    r_h, r_a, new_h, new_a = engine.update(home_team, away_team, home_goals, away_goals)\n",
    "    \n",
    "    home_elo_before.append(r_h)\n",
    "    away_elo_before.append(r_a)\n",
    "\n",
    "df_clean['HomeElo'] = home_elo_before\n",
    "df_clean['AwayElo'] = away_elo_before\n",
    "\n",
    "print(\"Engine ratings updated successfully with MoV.\")"
]

with open("notebooks/02_elo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated!")
