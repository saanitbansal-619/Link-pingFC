"""
League Analytics Dashboard — Tactical Analysis
Streamlit app: reads Master_Rivals_FeatureEngineered.csv (read-only).
Premium visual-first UI with dark theme. Data logic unchanged.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths and feature groups (single source of truth)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Master_Rivals_FeatureEngineered.csv")

# Attack = attacking + attacking transition (merged)
ATTACKING_FEATURES = [
    "Finishing_Efficiency", "Shot_Quality", "Box_Conversion",
    "Offensive_Duel_Win_Rate", "Cross_Accuracy",
    "High_Recovery_Rate", "Pressing_Intensity",
]
MIDFIELD_FEATURES = [
    "Verticality_Index", "Progressive_Pass_Rate", "Final_Third_Pass_Accuracy",
    "Possession_Control",
]
# Defence = defensive + defensive transition (merged)
DEFENSIVE_FEATURES = [
    "Defensive_Duel_Win_Rate", "Aerial_Duel_Win_Rate", "Shot_Suppression",
    "Shots_On_Target_Against_Rate",
    "High_Loss_Rate", "Def_Transition_Exposure",
]

# All engineered features across phases (for global insights)
ALL_FEATURES = ATTACKING_FEATURES + MIDFIELD_FEATURES + DEFENSIVE_FEATURES

# Internal phase key -> feature list (only 3 phases now)
PHASE_OPTIONS = {
    "Attacking": ATTACKING_FEATURES,
    "Midfield": MIDFIELD_FEATURES,
    "Defence": DEFENSIVE_FEATURES,
}

# Display order: Defence, Attack, Midfield (no separate transition phases)
PHASE_NAV_ORDER = ["Defence", "Attack", "Midfield"]
DISPLAY_TO_INTERNAL = {
    "Defence": "Defence",
    "Attack": "Attacking",
    "Midfield": "Midfield",
}

# Theme
BG_DARK = "#0B1C2D"
CARD_BG = "#13293D"
ACCENT_BLUE = "#3A86FF"
LEAGUE_GRAY = "#94A3B8"
COMPARISON_ORANGE = "#F59E0B"


def apply_custom_css():
    """Inject dark theme and hide default Streamlit menu/footer."""
    st.markdown(
        f"""
        <style>
        /* Global dark theme */
        .stApp {{
            background-color: {BG_DARK};
        }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Hide default Streamlit chrome */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}
        [data-testid="stToolbar"] {{ visibility: hidden; }}

        /* Typography hierarchy */
        h1 {{ color: #fff !important; font-size: 42px !important; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 0.35rem; }}
        h2 {{ color: #fff !important; font-size: 36px !important; font-weight: 600; margin: 1rem 0 0.35rem 0; }}
        h3 {{ color: #fff !important; font-size: 26px !important; font-weight: 600; margin-bottom: 0.75rem; }}
        h3.section-major {{ color: #fff !important; }}
        .subtitle-text {{ color: #fff; font-size: 20px; opacity: 0.85; margin-bottom: 0.5rem; }}
        .metric-value {{ font-size: 32px; font-weight: 700; color: #fff; margin-top: 0.25rem; }}
        .metric-label {{ font-size: 16px; opacity: 0.8; color: #fff; margin-top: 0.35rem; }}

        /* Spacing: major sections */
        .section-major {{ margin-bottom: 20px; }}
        .block-container {{ padding: 1.5rem 2rem 2rem 2rem; max-width: 100%%; padding-top: 2rem; }}
        [data-testid="stVerticalBlock"] > div {{ margin-bottom: 1rem; }}
        .stPlotlyChart {{ margin-bottom: 1.5rem; }}
        .element-container {{ margin-bottom: 0.5rem; }}

        /* Card styling */
        .tactical-card {{
            background: {CARD_BG};
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            border: 1px solid rgba(58, 134, 255, 0.15);
            margin-bottom: 1rem;
        }}
        .tactical-card .metric-name {{ color: #fff; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .tactical-card .metric-value {{ color: #fff; font-size: 32px; font-weight: 700; margin-top: 0.25rem; }}
        .tactical-card .metric-label {{ color: #fff; font-size: 16px; opacity: 0.8; margin-top: 0.35rem; }}

        /* Phase nav: white text on all buttons */
        .stRadio > div {{ flex-direction: row; gap: 0.5rem; flex-wrap: wrap; }}
        .stRadio label {{ background: {CARD_BG}; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid rgba(58,134,255,0.2); margin-bottom: 0.5rem; color: #fff !important; }}
        .stRadio label span {{ color: #fff !important; }}
        .stRadio label p {{ color: #fff !important; }}
        .stRadio label div {{ color: #fff !important; }}
        .stRadio label * {{ color: #fff !important; }}
        .stRadio label[data-checked="true"] {{ background: {ACCENT_BLUE}; border-color: {ACCENT_BLUE}; color: #fff !important; }}
        .stRadio label[data-checked="true"] span {{ color: #fff !important; }}
        .stRadio label[data-checked="true"] p {{ color: #fff !important; }}
        .stRadio label[data-checked="true"] * {{ color: #fff !important; }}

        /* Context / insights cards */
        .context-section {{ background: {CARD_BG}; border-radius: 12px; padding: 1.25rem; margin-top: 1rem; margin-bottom: 20px; border: 1px solid rgba(58, 134, 255, 0.15); box-shadow: 0 4px 14px rgba(0,0,0,0.25); }}
        .context-title {{ color: #fff; font-size: 26px; font-weight: 600; margin-bottom: 0.75rem; }}
        .context-list {{ margin: 0; padding-left: 1.25rem; color: #fff; font-size: 1rem; line-height: 1.6; }}
        .context-list strong {{ color: {ACCENT_BLUE}; }}
        .insight-above {{ color: #34D399; }}
        .insight-below {{ color: #fff; }}
        .insight-delta {{ color: #fff; font-size: 0.9em; opacity: 0.9; }}

        /* Tactical insights section */
        .tactical-insights {{ background: {CARD_BG}; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; margin-bottom: 20px; border: 1px solid rgba(58, 134, 255, 0.15); box-shadow: 0 4px 14px rgba(0,0,0,0.25); }}
        .tactical-insights-title {{ color: {ACCENT_BLUE}; font-size: 26px; font-weight: 700; margin-bottom: 1rem; }}
        .tactical-insight-item {{ color: #fff; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1rem; padding-left: 1.5rem; position: relative; }}
        .tactical-insight-item:before {{ content: "→"; position: absolute; left: 0; color: {ACCENT_BLUE}; font-weight: bold; }}
        .tactical-insight-item:last-child {{ margin-bottom: 0; }}

        /* Chart section headers (h3-level) */
        .chart-header {{ color: #ffffff !important; font-size: 26px; font-weight: 600; margin-bottom: 0.75rem; margin-top: 0.5rem; }}
        
        /* Ensure plotly charts have proper spacing */
        .js-plotly-plot {{ margin-bottom: 1rem; }}

        /* Selectbox and other widget labels: white */
        [data-testid="stSelectbox"] label {{ color: #fff !important; }}
        [data-testid="stSelectbox"] label p {{ color: #fff !important; }}
        .stSelectbox label {{ color: #fff !important; }}
        .stSelectbox label * {{ color: #fff !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    """
    Load CSV safely; handle file not found, strip column whitespace.
    Returns (df or None, error_message or None). Cached for performance.
    """
    if not os.path.isfile(DATA_PATH):
        return None, f"File not found: {DATA_PATH}"
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        return df, None
    except Exception as e:
        return None, str(e)


def get_phase_features(phase_display_name, columns):
    """Return only feature names that exist in the dataframe for the selected phase."""
    internal = DISPLAY_TO_INTERNAL.get(phase_display_name, phase_display_name)
    wanted = PHASE_OPTIONS.get(internal, [])
    return [f for f in wanted if f in columns]


def team_averages_by_phase(df, team, features):
    """Mean of each feature for the given team (over all matches)."""
    if not features or team not in df["Team"].values:
        return pd.Series(dtype=float)
    subset = df.loc[df["Team"] == team, features]
    return subset.mean()


def league_averages_by_phase(df, features):
    """Mean of each feature across all matches (league average)."""
    if not features:
        return pd.Series(dtype=float)
    return df[features].mean()


def get_league_rank(df, team, feature):
    """Return 1-based rank of team for given feature (higher = better). Total teams count for denominator."""
    if feature not in df.columns or "Team" not in df.columns:
        return None, 0
    team_means = df.groupby("Team")[feature].mean()
    total = len(team_means)
    if total == 0 or team not in team_means.index:
        return None, total
    rank = team_means.rank(ascending=False, method="min").loc[team]
    return int(rank), total


def build_ranking_table(df, features, sort_by_feature):
    """One row per team: means per feature, rank and percentile. Sorted by sort_by_feature (higher = better)."""
    if not features:
        return pd.DataFrame()
    team_means = df.groupby("Team")[features].mean().reset_index()
    sort_col = sort_by_feature if sort_by_feature in team_means.columns else features[0]
    team_means = team_means.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    total = len(team_means)
    team_means["Rank"] = range(1, total + 1)
    team_means["Percentile"] = (team_means["Rank"] / total).round(3)
    return team_means


def create_metric_cards(df, team, features, team_avg, total_teams):
    """
    Build 3–4 metric cards: icon, name, large value, small label (League Rank or Season Avg).
    Returns list of HTML strings, one per card (max 4).
    """
    cards = []
    metric_icons = ["📊", "⚡", "🎯", "📈"]
    for i, feat in enumerate(features[:4]):
        if feat not in team_avg.index:
            continue
        value = team_avg[feat]
        rank, total = get_league_rank(df, team, feat)
        if total and rank is not None:
            label = f"League Rank {rank}/{total}"
        else:
            label = "Season Avg"
        display_name = feat.replace("_", " ")
        icon = metric_icons[i % len(metric_icons)]
        if isinstance(value, (np.floating, float)):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)
        cards.append(
            f"""
            <div class="tactical-card">
                <div class="metric-name">{icon} {display_name}</div>
                <div class="metric-value">{value_str}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
        )
    return cards


def create_bar_chart(team_avg, league_avg, features, team_name, comparison_avg=None, comparison_name=None):
    """
    Bar chart: selected team vs league average; optional third series for comparison team.
    Team = accent blue, League Avg = light gray, Comparison = orange.
    """
    short_names = [f.replace("_", " ").replace("Rate", "").replace("Efficiency", "").strip()[:18] for f in features]
    metrics = short_names * 2
    values = list(team_avg) + list(league_avg)
    series = [team_name] * len(features) + ["League Avg"] * len(features)
    color_map = {team_name: ACCENT_BLUE, "League Avg": LEAGUE_GRAY}

    if comparison_avg is not None and comparison_name and len(comparison_avg):
        metrics = short_names * 3
        values = list(team_avg) + list(league_avg) + list(comparison_avg)
        series = [team_name] * len(features) + ["League Avg"] * len(features) + [comparison_name] * len(features)
        color_map[comparison_name] = COMPARISON_ORANGE

    plot_df = pd.DataFrame({
        "Metric": metrics,
        "Value": values,
        "Series": series,
    })
    fig = px.bar(
        plot_df, x="Metric", y="Value", color="Series",
        barmode="group",
        color_discrete_map=color_map,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff", size=11),
        xaxis=dict(
            tickangle=-40,
            gridcolor="rgba(148,163,184,0.2)",
            tickfont=dict(size=10, color="#ffffff"),
            automargin=True,
        ),
        yaxis=dict(gridcolor="rgba(148,163,184,0.2)", title="", tickfont=dict(color="#ffffff")),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(color="#ffffff", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=10, b=140, l=50, r=20),
        title=None,
        bargap=0.35,
        bargroupgap=0.1,
    )
    return fig


def create_radar_chart(team_avg, features, team_name, comparison_avg=None, comparison_name=None):
    """
    Radar chart for phase profile. Accent blue for selected team; optional second line (orange) for comparison.
    """
    r = list(team_avg)
    theta = list(features)
    r_closed = r + [r[0]]
    theta_closed = theta + [theta[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_closed, theta=theta_closed, fill="toself", name=team_name,
            line=dict(color=ACCENT_BLUE, width=2),
            fillcolor="rgba(58, 134, 255, 0.25)",
        )
    )
    if comparison_avg is not None and comparison_name and len(comparison_avg):
        r2 = list(comparison_avg) + [list(comparison_avg)[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=r2, theta=theta_closed, fill="toself", name=comparison_name,
                line=dict(color=COMPARISON_ORANGE, width=2),
                fillcolor="rgba(245, 158, 11, 0.15)",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title=None,
        polar=dict(
            bgcolor="rgba(19, 41, 61, 0.6)",
            domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(148,163,184,0.25)",
                tickfont=dict(color="#ffffff", size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.25)",
                tickfont=dict(color="#ffffff", size=10),
                layer="above traces",
            ),
        ),
        font=dict(color="#ffffff", size=11),
        showlegend=comparison_name is not None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(color="#ffffff", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20, b=80, l=60, r=60),
        height=400,
    )
    return fig


def create_context_section(team_avg, league_avg, features, team_name):
    """
    Build a 'Team in context' block: above vs below league average with deltas.
    Replaces raw table with coach-friendly narrative and context.
    """
    above, below = [], []
    for f in features:
        if f not in team_avg.index or f not in league_avg.index:
            continue
        t_val = team_avg[f]
        l_val = league_avg[f]
        name = f.replace("_", " ")
        delta = t_val - l_val
        if l_val != 0 and not np.isnan(l_val):
            pct = (delta / abs(l_val)) * 100
            delta_str = f" ({pct:+.0f}% vs league)"
        else:
            delta_str = ""
        if delta > 0:
            above.append((name, t_val, l_val, delta_str))
        elif delta < 0:
            below.append((name, t_val, l_val, delta_str))
        # delta == 0: skip or add to "in line"; keep it simple, skip

    lines = []
    if above:
        lines.append(f'<div class="context-title">↑ Above league average</div>')
        lines.append('<ul class="context-list">')
        for name, t_val, l_val, d in above:
            lines.append(f'<li class="insight-above"><strong>{name}</strong> — {t_val:.3f} vs league {l_val:.3f}<span class="insight-delta">{d}</span></li>')
        lines.append('</ul>')
    if below:
        lines.append(f'<div class="context-title">↓ Below league average</div>')
        lines.append('<ul class="context-list">')
        for name, t_val, l_val, d in below:
            lines.append(f'<li class="insight-below"><strong>{name}</strong> — {t_val:.3f} vs league {l_val:.3f}<span class="insight-delta">{d}</span></li>')
        lines.append('</ul>')
    if not above and not below:
        lines.append('<p class="context-list">Metrics in line with league average for this phase.</p>')
    return '<div class="context-section">' + '\n'.join(lines) + '</div>'


def generate_tactical_insights(team_avg, league_avg, features, rival_name):
    """
    Generate 4 tactical insights for YOUR team: how to exploit this rival overall
    (attack, defence, midfield combined), not just the current phase.
    """
    # Find rival's WEAKNESSES (metrics where rival is below league average)
    weaknesses = []
    for f in features:
        if f not in team_avg.index or f not in league_avg.index:
            continue
        t_val = team_avg[f]  # rival's value
        l_val = league_avg[f]
        if pd.isna(t_val) or pd.isna(l_val) or l_val == 0:
            continue
        delta = t_val - l_val
        pct_below = (delta / abs(l_val)) * 100 if l_val != 0 else 0
        if delta < 0:  # rival is worse than league
            weaknesses.append((f, t_val, l_val, pct_below, delta))

    # Sort by how much they're below (biggest weakness first)
    weaknesses.sort(key=lambda x: x[3])  # most negative first

    insights = []

    # Rival weakness -> recommendation for YOUR team to exploit it
    exploit_map = {
        "Finishing_Efficiency": "They convert chances poorly — press high and force rushed shots, then counter quickly when they miss.",
        "Shot_Quality": "They struggle to create good chances — keep a compact block and deny space in the box; they will take low-quality shots.",
        "Box_Conversion": "They are inefficient in the penalty area — get bodies in the box when you attack; their defence will crack under sustained pressure.",
        "Offensive_Duel_Win_Rate": "They lose offensive duels often — encourage 1v1s in wide areas and double up when they receive in dangerous positions.",
        "Cross_Accuracy": "Their crossing is below average — force them wide, stay compact in the box, and win first contacts from their crosses.",
        "Verticality_Index": "They play few forward passes — press their midfield and force them long; they will struggle to play through you.",
        "Progressive_Pass_Rate": "They rarely progress the ball — press aggressively in midfield; they will turn over possession in dangerous areas.",
        "Final_Third_Pass_Accuracy": "They are sloppy in the final third — stay compact, cut passing lanes, and spring counter-attacks when they give the ball away.",
        "Possession_Control": "They lose the ball often — press in groups to force mistakes and win the ball high up the pitch.",
        "Defensive_Duel_Win_Rate": "They lose defensive duels — run at their back line, commit players forward, and exploit 1v1s in the final third.",
        "Aerial_Duel_Win_Rate": "They are weak in the air — use set-pieces and long balls into the box; target their weakest aerial defender.",
        "Shot_Suppression": "They allow lots of shots — be bold in attack; you will get opportunities if you commit numbers forward.",
        "Shots_On_Target_Against_Rate": "Many shots against them go on target — take shots when in range and get runners in the box for rebounds.",
        "High_Recovery_Rate": "They rarely win the ball high up — build from the back calmly; they will not hurt you with a high press.",
        "Pressing_Intensity": "Their press is weak — play out from the back and through midfield; you will have time on the ball.",
        "High_Loss_Rate": "They lose the ball in dangerous areas — set pressing traps and pounce on loose balls to create quick chances.",
        "Def_Transition_Exposure": "They are exposed when they lose the ball — counter at speed immediately after regaining possession.",
    }

    used_features = set()
    for feat, t_val, l_val, pct, delta in weaknesses[:10]:
        if feat in exploit_map and feat not in used_features:
            insights.append(exploit_map[feat])
            used_features.add(feat)
            if len(insights) >= 4:
                break

    # Fallbacks: general ways to exploit rivals, across all aspects
    if len(insights) < 4:
        general_exploits = [
            "Target their weakest defenders 1v1 and isolate them in wide areas.",
            "Increase tempo after turnovers — many rivals struggle to recover their shape quickly.",
            "Use set-pieces aggressively — crosses, second balls and rehearsed routines will create chances.",
            "Control the ball in midfield, then switch play quickly to attack spaces on the far side.",
            "Be brave playing out from the back when they press — they leave gaps between the lines.",
        ]
        for g in general_exploits:
            if len(insights) < 4:
                insights.append(g)

    while len(insights) < 4:
        insights.append("Stay disciplined and take your chances — consistency against every rival wins the league.")

    html_lines = [
        '<div class="tactical-insights">',
        '<div class="tactical-insights-title">🎯 Tactical insights</div>',
    ]
    for insight in insights[:4]:
        html_lines.append(f'<div class="tactical-insight-item">{insight}</div>')
    html_lines.append('</div>')

    return '\n'.join(html_lines)


def main():
    st.set_page_config(layout="wide", page_title="League Analytics Dashboard")
    apply_custom_css()

    # Load data
    df, err = load_data()
    if err is not None:
        st.error(err)
        return
    if df is None or df.empty:
        st.warning("No data loaded.")
        return

    teams = sorted(df["Team"].dropna().unique().tolist())
    if not teams:
        st.warning("No teams in data.")
        return

    # -----------------------------------------------------------------------
    # Header: title + subtitle; team selector on right
    # -----------------------------------------------------------------------
    col_title, col_team = st.columns([3, 1])
    with col_title:
        st.markdown("""
        <h1>League Analytics Dashboard</h1>
        <p class='subtitle-text'>Comprehensive Tactical Analysis</p>
        """, unsafe_allow_html=True)
    with col_team:
        team_selected = st.selectbox("Team", options=teams, key="team_select", label_visibility="collapsed")

    st.markdown(f"""
    <h2>{team_selected}</h2>
    <p class='subtitle-text'>Tactical Analysis Dashboard</p>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Phase navigation (horizontal)
    # -----------------------------------------------------------------------
    phase_selected = st.radio(
        "Phase",
        options=PHASE_NAV_ORDER,
        key="phase_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    internal_phase = DISPLAY_TO_INTERNAL.get(phase_selected, phase_selected)
    features = get_phase_features(phase_selected, df.columns)
    if not features:
        st.info(f"No engineered features available for **{phase_selected}**. Check CSV columns.")
        return

    team_avg = team_averages_by_phase(df, team_selected, features)
    league_avg = league_averages_by_phase(df, features)
    total_teams = df["Team"].nunique()

    # Optional comparison (second radar line)
    compare_label = st.selectbox(
        "Compare with team (optional)",
        options=["— None —"] + [t for t in teams if t != team_selected],
        key="compare_team",
    )
    comparison_team = None if compare_label == "— None —" else compare_label
    comparison_avg = team_averages_by_phase(df, comparison_team, features) if comparison_team else None

    # -----------------------------------------------------------------------
    # Metric cards row (3–4 cards)
    # -----------------------------------------------------------------------
    cards_html = create_metric_cards(df, team_selected, features, team_avg, total_teams)
    n_cards = len(cards_html)
    if n_cards:
        cols = st.columns(min(n_cards, 4))
        for i, html in enumerate(cards_html[:4]):
            with cols[i % 4]:
                st.markdown(html, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Two-column: Bar chart (left), Radar (right)
    # -----------------------------------------------------------------------
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    col_bar, col_radar = st.columns(2)
    with col_bar:
        st.markdown('<p class="chart-header">Team vs League Average</p>', unsafe_allow_html=True)
        fig_bar = create_bar_chart(
            team_avg, league_avg, features, team_selected,
            comparison_avg=comparison_avg, comparison_name=comparison_team,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_radar:
        st.markdown('<p class="chart-header">Phase profile (radar)</p>', unsafe_allow_html=True)
        fig_radar = create_radar_chart(
            team_avg, features, team_selected,
            comparison_avg=comparison_avg, comparison_name=comparison_team,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # -----------------------------------------------------------------------
    # Team in context (replaces table — narrative + above/below league)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<h3 class="section-major">Where this team stands</h3>', unsafe_allow_html=True)
    context_html = create_context_section(team_avg, league_avg, features, team_selected)
    st.markdown(context_html, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Tactical insights (4 actionable points, overall team profile)
    # -----------------------------------------------------------------------
    all_feats = [f for f in ALL_FEATURES if f in df.columns]
    team_avg_all = team_averages_by_phase(df, team_selected, all_feats)
    league_avg_all = league_averages_by_phase(df, all_feats)
    insights_html = generate_tactical_insights(team_avg_all, league_avg_all, all_feats, team_selected)
    st.markdown(insights_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    st.write("Dashboard ready.")
