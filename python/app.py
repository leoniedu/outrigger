#!/usr/bin/env python3
"""Streamlit web app for the outrigger rotation optimizer."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from outrigger_opt import solve_rotation_cycle
from outrigger_opt.meta import optimize_stint_range


def analyze_manual_schedule(schedule_df, paddler_weights, n_seats):
    """Analyze a manually entered schedule for trim and MOI.

    Args:
        schedule_df: DataFrame with columns for each seat, rows for each stint.
                    Cell values are paddler names.
        paddler_weights: Dict mapping paddler name to weight in kg.
        n_seats: Number of seats.

    Returns:
        dict with trim_stats similar to solver output.
    """
    # Seat positions (meters from center, negative = bow)
    seat_positions = [-(n_seats-1)/2 + i for i in range(n_seats)]
    sum_pos_sq = sum(p**2 for p in seat_positions)
    avg_paddler_weight = sum(paddler_weights.values()) / len(paddler_weights)

    cycle_length = len(schedule_df)
    trim_moments = []
    moi_values = []
    seat_breakdown = []

    for stint_idx in range(cycle_length):
        stint_trim = 0.0
        stint_moi = 0.0
        stint_seats = []

        for seat_idx in range(n_seats):
            col_name = schedule_df.columns[seat_idx]
            paddler_name = schedule_df.iloc[stint_idx, seat_idx]
            weight = paddler_weights.get(paddler_name, avg_paddler_weight)
            position = seat_positions[seat_idx]

            trim_contrib = weight * position
            moi_contrib = weight * (position ** 2)

            stint_trim += trim_contrib
            stint_moi += moi_contrib

            stint_seats.append({
                'seat': seat_idx + 1,
                'name': paddler_name,
                'weight': weight,
                'position': position,
                'trim_contrib': trim_contrib,
                'moi_contrib': moi_contrib,
            })

        trim_moments.append(stint_trim)
        moi_values.append(stint_moi)
        seat_breakdown.append(stint_seats)

    return {
        'trim_moments': trim_moments,
        'max_abs_trim_moment': max(abs(m) for m in trim_moments),
        'moi_values': moi_values,
        'avg_moi': sum(moi_values) / cycle_length,
        'seat_positions': seat_positions,
        'seat_breakdown': seat_breakdown,
        'avg_paddler_weight': avg_paddler_weight,
        'normalized_max_abs_trim': max(abs(m) for m in trim_moments) / (avg_paddler_weight * 2.5),
        'normalized_avg_moi': sum(moi_values) / cycle_length / (avg_paddler_weight * sum_pos_sq),
    }


def generate_cycle_rules(schedule_df, all_paddlers, n_seats):
    """Generate rotation rules from a schedule (similar to solver output).

    Args:
        schedule_df: DataFrame with columns for each seat, rows for each stint.
        all_paddlers: List of all paddler names.
        n_seats: Number of seats.

    Returns:
        dict mapping paddler name to rule string (e.g., "1→2→3→Out")
    """
    cycle_length = len(schedule_df)
    rules = {}

    for paddler in all_paddlers:
        positions = []
        for stint_idx in range(cycle_length):
            # Find which seat this paddler is in (or Out)
            found = False
            for seat_idx in range(n_seats):
                if schedule_df.iloc[stint_idx, seat_idx] == paddler:
                    positions.append(str(seat_idx + 1))
                    found = True
                    break
            if not found:
                positions.append("Out")
        rules[paddler] = "→".join(positions)

    return rules


def generate_paddler_summary(schedule_df, all_paddlers, n_seats):
    """Generate paddler summary from a schedule.

    Args:
        schedule_df: DataFrame with columns for each seat, rows for each stint.
        all_paddlers: List of all paddler names.
        n_seats: Number of seats.

    Returns:
        DataFrame with paddler statistics.
    """
    cycle_length = len(schedule_df)
    summary_data = []

    for paddler in all_paddlers:
        stints_paddled = 0
        stints_rested = 0
        current_stretch = 0
        longest_stretch = 0

        for stint_idx in range(cycle_length):
            # Check if paddler is in any seat
            in_canoe = False
            for seat_idx in range(n_seats):
                if schedule_df.iloc[stint_idx, seat_idx] == paddler:
                    in_canoe = True
                    break

            if in_canoe:
                stints_paddled += 1
                current_stretch += 1
                longest_stretch = max(longest_stretch, current_stretch)
            else:
                stints_rested += 1
                current_stretch = 0

        summary_data.append({
            'name': paddler,
            'stints_paddled': stints_paddled,
            'stints_rested': stints_rested,
            'longest_stretch_stints': longest_stretch,
            'stints_paddled_per_cycle': stints_paddled,
        })

    return pd.DataFrame(summary_data)


# Translations
TRANSLATIONS = {
    "en": {
        "page_title": "Outrigger Rotation Optimizer",
        "title": "Outrigger Rotation Optimizer",
        "subtitle": "Optimize crew rotation schedules to minimize race time while managing fatigue.",
        "language": "Language",
        "race_parameters": "Race Parameters",
        "distance_km": "Distance (km)",
        "base_speed": "Base Speed (km/h)",
        "stint_distance": "Stint Distance (km)",
        "stint_distance_help": "Distance to paddle in each stint before rotation",
        "expected_stint_time": "Expected Stint Time",
        "at_base_speed": "(at base speed)",
        "switch_time": "Switch Time (s)",
        "max_consecutive": "Max Consecutive Stints",
        "max_consecutive_help": "Maximum stints a paddler can paddle in a row",
        "seat_assignment_params": "Seat Assignment",
        "seat_assignment_note": "These affect who sits where (requires re-optimization)",
        "stint_time_params": "Race Simulation",
        "stint_time_note": "These only affect timing, not seat assignments",
        "fatigue_params": "Fatigue Model",
        "fatigue_work_rate": "Work Rate",
        "fatigue_work_rate_help": "W' depletion per minute (0.015 = 15% per 10 min)",
        "fatigue_tau_recovery": "Recovery τ (min)",
        "fatigue_tau_recovery_help": "Recovery time constant (7 min ≈ 5 min half-life)",
        "power_speed_exponent": "Power-Speed Exponent",
        "power_speed_exponent_help": "Speed = power^exp (0.4 for water drag, 1.0 for linear)",
        "crew_configuration": "Crew Configuration",
        "number_of_seats": "Number of Seats",
        "paddlers_resting": "Paddlers Resting per Stint",
        "solver_settings": "Solver Settings",
        "time_limit": "Time Limit (seconds)",
        "gap_tolerance": "Gap Tolerance",
        "balance_penalties": "Balance Penalties",
        "balance_explanation": """**Trim** measures fore-aft balance. Positive = stern-heavy, negative = bow-heavy. Keep close to zero.

**MOI (Moment of Inertia)** measures weight distribution along the canoe's length. Low MOI = weight in the middle; High MOI = weight at the ends.

| Condition | Preferred MOI |
|-----------|---------------|
| Downwind/runners | Low (more maneuverable) |
| Flat water race | Low |
| Rough crosswind/chop | High (more stable) |
| Strong side swells | High |

**Default:** Most races favor low MOI for responsiveness.""",
        "trim_penalty": "Trim Penalty",
        "trim_penalty_help": "How much to penalize trim imbalance (0-1 scale). 1.0 = ~10% output cost for typical worst-case trim. 0 = disabled.",
        "moi_penalty": "MOI Penalty",
        "moi_penalty_help": "How much to penalize weight distribution (-1 to 1). 1.0 = ~12% output cost for typical MOI. Negative = prefer weight at ends (more stable).",
        "steerer_paddle_fraction": "Steerer Paddle %",
        "steerer_paddle_fraction_help": "Fraction of time steerer paddles vs steers. 70-80% flat, 50-60% moderate, 30-40% rough water. Affects output and dead weight penalty.",
        "normalized": "Normalized",
        "crew_data": "Crew Data",
        "upload_csv": "Upload crew CSV (optional)",
        "upload_csv_help": "CSV with columns: name, ability, weight, seat1, seat2, ..., seat6",
        "loaded_paddlers": "Loaded {n} paddlers from uploaded file",
        "paddlers": "Paddlers",
        "name": "Name",
        "ability": "Ability",
        "weight_kg": "Weight (kg)",
        "seat_configuration": "Seat Configuration",
        "seat_weights": "Seat Weights",
        "seat_weights_desc": "Importance weight for each seat (higher = more important)",
        "seat_entry_weights": "Seat Entry Weights",
        "seat_entry_desc": "How easy to enter each seat (>1 easier, <1 harder)",
        "seat": "Seat",
        "seat_eligibility": "Seat Eligibility",
        "seat_eligibility_desc": "Check the seats each paddler can occupy",
        "optimize": "Optimize",
        "run_optimization": "Run Optimization",
        "provide_paddlers_error": "Please provide exactly {n} paddler names.",
        "solving": "Solving optimization problem...",
        "optimization_failed": "Optimization failed: {error}",
        "results": "Results",
        "status": "Status",
        "race_time": "Race Time",
        "avg_output": "Avg Output",
        "stints": "Stints",
        "rotation_rules": "Rotation Rules",
        "rotation_rules_desc": "Each paddler follows this repeating pattern:",
        "paddler": "Paddler",
        "rule": "Rule",
        "cycle_schedule": "Cycle Schedule",
        "cycle_schedule_desc": "This {n}-stint cycle repeats throughout the race:",
        "stint": "Stint",
        "paddler_summary": "Paddler Summary",
        "balance_analysis": "Balance Analysis",
        "avg_trim": "Avg Trim (abs)",
        "max_trim": "Max Trim (abs)",
        "avg_moi": "Avg MOI",
        "cycle_stint_details": "Cycle Stint Details",
        "trim_kgm": "Trim (kg-m)",
        "direction": "Direction",
        "moi_kgm2": "MOI (kg-m²)",
        "avg_output": "Avg Output",
        "stern": "stern",
        "bow": "bow",
        "neutral": "neutral",
        "seat_breakdown": "Seat Breakdown (Trim & MOI Calculation)",
        "seat": "Seat",
        "weight_kg": "Weight (kg)",
        "position_m": "Position (m)",
        "trim_contrib": "Trim (kg·m)",
        "moi_contrib": "MOI (kg·m²)",
        "total": "Total",
        "aggregate_stats": "Aggregate Stats",
        "metric": "Metric",
        "value": "Value",
        "cycle_length": "Cycle Length",
        "total_stints": "Total Stints",
        "avg_time_per_paddler": "Avg Time per Paddler",
        "max_time_any": "Max Time (any paddler)",
        "min_time_any": "Min Time (any paddler)",
        "max_consecutive_stretch": "Max Consecutive Stretch",
        "race_summary": "Race Summary",
        "distance": "Distance",
        "avg_stint_time": "Avg Stint Time",
        "number_of_switches": "Number of Switches",
        "effective_speed": "Effective Speed",
        "rotation_pattern": "Rotation Pattern",
        "rotation_pattern_desc": "Cycle of {n} stints repeats {times}x throughout the race",
        "out": "Out",
        "footer": "Powered by PuLP/CBC mixed-integer programming solver",
        "interface_mode": "Interface Mode",
        "mode_simple": "Seat Assignment Only",
        "mode_simple_desc": "Optimize rotation pattern without race timing",
        "mode_full": "Full (with Race Time)",
        "mode_full_desc": "Include race simulation and stint optimization",
        "mode_manual": "Manual Analysis",
        "mode_manual_desc": "Enter your own rotation schedule and analyze trim/MOI",
        "manual_schedule": "Manual Schedule",
        "manual_schedule_help": "Enter paddler names for each seat in each stint. Use the paddler names from the crew list.",
        "analyze_schedule": "Analyze Schedule",
        "analyzing": "Analyzing schedule...",
        "manual_analysis_results": "Manual Schedule Analysis",
        "invalid_paddler": "Invalid paddler name '{name}' in Stint {stint}, Seat {seat}",
        "duplicate_paddler": "Paddler '{name}' appears twice in Stint {stint}",
        "cycle_stints": "Cycle Stints",
        "optimize_stint_length": "Optimize Stint Length",
        "optimize_stint_help": "Search for best stint distance (runs multiple optimizations)",
        "stint_range_min": "Min Stint (km)",
        "stint_range_max": "Max Stint (km)",
        "stint_range_step": "Step (km)",
        "stint_optimization_results": "Stint Length Optimization",
        "best_stint": "Best Stint Distance",
        "stint_comparison": "Stint Comparison",
    },
    "pt": {
        "page_title": "Otimizador de Rotação de Canoa",
        "title": "Otimizador de Rotação de Canoa",
        "subtitle": "Otimize escalas de rotação da tripulação para minimizar o tempo de prova gerenciando a fadiga.",
        "language": "Idioma",
        "race_parameters": "Parâmetros da Prova",
        "distance_km": "Distância (km)",
        "base_speed": "Velocidade Base (km/h)",
        "stint_distance": "Distância por Turno (km)",
        "stint_distance_help": "Distância a remar em cada turno antes da rotação",
        "expected_stint_time": "Tempo Esperado do Turno",
        "at_base_speed": "(na velocidade base)",
        "switch_time": "Tempo de Troca (s)",
        "max_consecutive": "Turnos Consecutivos Máx.",
        "max_consecutive_help": "Máximo de turnos que um remador pode remar seguidos",
        "seat_assignment_params": "Atribuição de Bancos",
        "seat_assignment_note": "Afetam quem senta onde (requer re-otimização)",
        "stint_time_params": "Simulação da Prova",
        "stint_time_note": "Afetam apenas o tempo, não a atribuição de bancos",
        "fatigue_params": "Modelo de Fadiga",
        "fatigue_work_rate": "Taxa de Trabalho",
        "fatigue_work_rate_help": "Depleção de W' por minuto (0.015 = 15% por 10 min)",
        "fatigue_tau_recovery": "τ de Recuperação (min)",
        "fatigue_tau_recovery_help": "Constante de tempo de recuperação (7 min ≈ 5 min meia-vida)",
        "power_speed_exponent": "Expoente Potência-Velocidade",
        "power_speed_exponent_help": "Velocidade = potência^exp (0.4 para arrasto na água, 1.0 linear)",
        "crew_configuration": "Configuração da Tripulação",
        "number_of_seats": "Número de Bancos",
        "paddlers_resting": "Remadores Descansando por Turno",
        "solver_settings": "Configurações do Solver",
        "time_limit": "Limite de Tempo (segundos)",
        "gap_tolerance": "Tolerância de Gap",
        "balance_penalties": "Penalidades de Equilíbrio",
        "balance_explanation": """**Trim** mede o equilíbrio proa-popa. Positivo = peso na popa, negativo = peso na proa. Mantenha próximo de zero.

**MOI (Momento de Inércia)** mede a distribuição de peso ao longo da canoa. MOI baixo = peso no centro; MOI alto = peso nas extremidades.

| Condição | MOI Preferido |
|----------|---------------|
| Downwind/surfando ondas | Baixo (mais manobrável) |
| Água calma | Baixo |
| Mar agitado/ondas laterais | Alto (mais estável) |
| Swell lateral forte | Alto |

**Padrão:** A maioria das provas favorece MOI baixo para maior resposta ao leme.""",
        "trim_penalty": "Penalidade de Trim",
        "trim_penalty_help": "Quanto penalizar desequilíbrio de trim (escala 0-1). 1.0 = ~10% custo para trim típico. 0 = desativado.",
        "moi_penalty": "Penalidade de MOI",
        "moi_penalty_help": "Quanto penalizar distribuição de peso (-1 a 1). 1.0 = ~12% custo para MOI típico. Negativo = preferir peso nas extremidades (mais estável).",
        "steerer_paddle_fraction": "% Remada do Leme",
        "steerer_paddle_fraction_help": "Fração do tempo que o leme rema vs governa. 70-80% água calma, 50-60% moderado, 30-40% mar agitado. Afeta contribuição e peso morto.",
        "normalized": "Normalizado",
        "crew_data": "Dados da Tripulação",
        "upload_csv": "Carregar CSV da tripulação (opcional)",
        "upload_csv_help": "CSV com colunas: name, ability, weight, seat1, seat2, ..., seat6",
        "loaded_paddlers": "Carregados {n} remadores do arquivo",
        "paddlers": "Remadores",
        "name": "Nome",
        "ability": "Habilidade",
        "weight_kg": "Peso (kg)",
        "seat_configuration": "Configuração dos Bancos",
        "seat_weights": "Pesos dos Bancos",
        "seat_weights_desc": "Peso de importância para cada banco (maior = mais importante)",
        "seat_entry_weights": "Pesos de Entrada nos Bancos",
        "seat_entry_desc": "Facilidade de entrar em cada banco (>1 mais fácil, <1 mais difícil)",
        "seat": "Banco",
        "seat_eligibility": "Elegibilidade de Bancos",
        "seat_eligibility_desc": "Marque os bancos que cada remador pode ocupar",
        "optimize": "Otimizar",
        "run_optimization": "Executar Otimização",
        "provide_paddlers_error": "Por favor, forneça exatamente {n} nomes de remadores.",
        "solving": "Resolvendo problema de otimização...",
        "optimization_failed": "Otimização falhou: {error}",
        "results": "Resultados",
        "status": "Status",
        "race_time": "Tempo de Prova",
        "avg_output": "Saída Média",
        "stints": "Turnos",
        "rotation_rules": "Regras de Rotação",
        "rotation_rules_desc": "Cada remador segue este padrão repetitivo:",
        "paddler": "Remador",
        "rule": "Regra",
        "cycle_schedule": "Escala do Ciclo",
        "cycle_schedule_desc": "Este ciclo de {n} turnos se repete durante toda a prova:",
        "stint": "Turno",
        "paddler_summary": "Resumo dos Remadores",
        "balance_analysis": "Análise de Equilíbrio",
        "avg_trim": "Trim Médio (abs)",
        "max_trim": "Trim Máximo (abs)",
        "avg_moi": "MOI Médio",
        "cycle_stint_details": "Detalhes do Ciclo por Turno",
        "trim_kgm": "Trim (kg-m)",
        "direction": "Direção",
        "moi_kgm2": "MOI (kg-m²)",
        "avg_output": "Potência Média",
        "stern": "popa",
        "bow": "proa",
        "neutral": "neutro",
        "seat_breakdown": "Detalhamento por Banco (Cálculo Trim & MOI)",
        "seat": "Banco",
        "weight_kg": "Peso (kg)",
        "position_m": "Posição (m)",
        "trim_contrib": "Trim (kg·m)",
        "moi_contrib": "MOI (kg·m²)",
        "total": "Total",
        "aggregate_stats": "Estatísticas Agregadas",
        "metric": "Métrica",
        "value": "Valor",
        "cycle_length": "Duração do Ciclo",
        "total_stints": "Total de Turnos",
        "avg_time_per_paddler": "Tempo Médio por Remador",
        "max_time_any": "Tempo Máx. (qualquer remador)",
        "min_time_any": "Tempo Mín. (qualquer remador)",
        "max_consecutive_stretch": "Sequência Consecutiva Máx.",
        "race_summary": "Resumo da Prova",
        "distance": "Distância",
        "avg_stint_time": "Tempo Médio do Turno",
        "number_of_switches": "Número de Trocas",
        "effective_speed": "Velocidade Efetiva",
        "rotation_pattern": "Padrão de Rotação",
        "rotation_pattern_desc": "Ciclo de {n} turnos repete {times}x durante a prova",
        "out": "Fora",
        "footer": "Desenvolvido com PuLP/CBC mixed-integer programming solver",
        "interface_mode": "Modo de Interface",
        "mode_simple": "Apenas Atribuição de Bancos",
        "mode_simple_desc": "Otimizar padrão de rotação sem tempo de prova",
        "mode_full": "Completo (com Tempo de Prova)",
        "mode_full_desc": "Incluir simulação de prova e otimização de turno",
        "mode_manual": "Análise Manual",
        "mode_manual_desc": "Insira seu próprio esquema de rotação e analise trim/MOI",
        "manual_schedule": "Escala Manual",
        "manual_schedule_help": "Insira os nomes dos remadores para cada banco em cada turno. Use os nomes da lista de tripulação.",
        "analyze_schedule": "Analisar Escala",
        "analyzing": "Analisando escala...",
        "manual_analysis_results": "Análise da Escala Manual",
        "invalid_paddler": "Nome de remador inválido '{name}' no Turno {stint}, Banco {seat}",
        "duplicate_paddler": "Remador '{name}' aparece duas vezes no Turno {stint}",
        "cycle_stints": "Turnos do Ciclo",
        "optimize_stint_length": "Otimizar Duração do Turno",
        "optimize_stint_help": "Buscar melhor distância de turno (executa múltiplas otimizações)",
        "stint_range_min": "Turno Mín. (km)",
        "stint_range_max": "Turno Máx. (km)",
        "stint_range_step": "Passo (km)",
        "stint_optimization_results": "Otimização de Duração do Turno",
        "best_stint": "Melhor Distância de Turno",
        "stint_comparison": "Comparação de Turnos",
    }
}


def t(key):
    """Get translation for key in current language."""
    lang = st.session_state.get('language', 'pt')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)


st.set_page_config(
    page_title="Outrigger Rotation Optimizer",
    page_icon="🚣",
    layout="wide"
)

# Language selector at top of sidebar
if 'language' not in st.session_state:
    st.session_state.language = 'pt'

language = st.sidebar.selectbox(
    "🌐 Language / Idioma",
    options=['pt', 'en'],
    format_func=lambda x: 'Português' if x == 'pt' else 'English',
    index=0 if st.session_state.language == 'pt' else 1,
    key='language'
)

st.title(t("title"))
st.markdown(t("subtitle"))

# Default CSV file path
DEFAULT_CSV_PATH = Path(__file__).parent / "defaults_crew.csv"


def load_crew_defaults(csv_path=None, uploaded_file=None):
    """Load crew defaults from CSV file.

    Returns:
        tuple: (paddler_df, ability_list, weight_list, eligibility_matrix)
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return None, None, None, None

    # Extract data from CSV
    names = df['name'].tolist()
    ability = df['ability'].tolist()
    weight = df['weight'].tolist()

    # Extract eligibility matrix (columns seat1, seat2, ..., seat6)
    seat_cols = [col for col in df.columns if col.startswith('seat')]
    eligibility = df[seat_cols].values.astype(int)

    paddlers_df = pd.DataFrame({'name': names})
    return paddlers_df, ability, weight, eligibility


# Sidebar for parameters

# --- Interface Mode Selector ---
st.sidebar.header(t("interface_mode"))
interface_mode = st.sidebar.radio(
    t("interface_mode"),
    options=["simple", "full", "manual"],
    format_func=lambda x: {"simple": t("mode_simple"), "full": t("mode_full"), "manual": t("mode_manual")}[x],
    index=0,
    label_visibility="collapsed",
    help=t("mode_simple_desc")
)

# --- Crew Configuration (structural) ---
st.sidebar.markdown("---")
st.sidebar.header(t("crew_configuration"))
n_seats = st.sidebar.number_input(
    t("number_of_seats"),
    min_value=2, max_value=12, value=6, step=1
)
n_resting = st.sidebar.number_input(
    t("paddlers_resting"),
    min_value=1, max_value=10, value=3, step=1
)

# --- Seat Assignment Parameters (affect MIP optimization) ---
st.sidebar.markdown("---")
st.sidebar.header(t("seat_assignment_params"))

max_consecutive = st.sidebar.number_input(
    t("max_consecutive"),
    min_value=1, max_value=10, value=3, step=1,
    help=t("max_consecutive_help")
)

with st.sidebar.expander(t("balance_penalties"), expanded=False):
    st.markdown(t("balance_explanation"))
    # UI uses 0-1 scale; internally scaled to physical units (×0.001 for trim, ×0.0001 for MOI)
    trim_penalty_ui = st.number_input(
        t("trim_penalty"),
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help=t("trim_penalty_help")
    )
    moi_penalty_ui = st.number_input(
        t("moi_penalty"),
        min_value=-1.0, max_value=1.0, value=0.5, step=0.05,
        help=t("moi_penalty_help")
    )
    # Scale to physical units: trim 1.0 → 0.001 (0.1% per kg-m), moi 1.0 → 0.0001 (0.01% per kg-m²)
    trim_penalty_weight = trim_penalty_ui * 0.001
    moi_penalty_weight = moi_penalty_ui * 0.0001
    steerer_paddle_fraction = st.number_input(
        t("steerer_paddle_fraction"),
        min_value=0.2, max_value=1.0, value=0.6, step=0.05,
        help=t("steerer_paddle_fraction_help")
    )

# --- Race Simulation Parameters (only in full mode) ---
if interface_mode == "full":
    st.sidebar.markdown("---")
    st.sidebar.header(t("stint_time_params"))
    st.sidebar.caption(t("stint_time_note"))

    distance_km = st.sidebar.number_input(
        t("distance_km"),
        min_value=1.0, max_value=200.0, value=60.0, step=1.0
    )

    # Stint length optimization option
    optimize_stint = st.sidebar.checkbox(
        t("optimize_stint_length"),
        value=False,
        help=t("optimize_stint_help")
    )

    if optimize_stint:
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            stint_min = st.number_input(t("stint_range_min"), min_value=0.5, max_value=5.0, value=1.5, step=0.5)
        with col2:
            stint_max = st.number_input(t("stint_range_max"), min_value=1.0, max_value=10.0, value=4.0, step=0.5)
        with col3:
            stint_step = st.number_input(t("stint_range_step"), min_value=0.25, max_value=1.0, value=0.5, step=0.25)
        stint_km = 2.0  # default, will be optimized
    else:
        stint_km = st.sidebar.number_input(
            t("stint_distance"),
            min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            help=t("stint_distance_help")
        )
        stint_min, stint_max, stint_step = None, None, None

    speed_kmh = st.sidebar.number_input(
        t("base_speed"),
        min_value=1.0, max_value=20.0, value=10.0, step=0.5
    )

    # Calculate and display expected stint time (non-editable)
    expected_stint_min = (stint_km / speed_kmh) * 60
    if not optimize_stint:
        st.sidebar.markdown(
            f"**{t('expected_stint_time')}:** {expected_stint_min:.1f} min "
            f"<span style='color: gray; font-size: 0.85em;'>{t('at_base_speed')}</span>",
            unsafe_allow_html=True
        )

    switch_time_secs = st.sidebar.number_input(
        t("switch_time"),
        min_value=0, max_value=300, value=40, step=10
    )

    with st.sidebar.expander(t("fatigue_params"), expanded=False):
        fatigue_work_rate = st.number_input(
            t("fatigue_work_rate"),
            min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f",
            help=t("fatigue_work_rate_help")
        )
        fatigue_tau_recovery = st.number_input(
            t("fatigue_tau_recovery"),
            min_value=1.0, max_value=20.0, value=7.0, step=0.5,
            help=t("fatigue_tau_recovery_help")
        )
        power_speed_exponent = st.number_input(
            t("power_speed_exponent"),
            min_value=0.1, max_value=1.0, value=0.4, step=0.05,
            help=t("power_speed_exponent_help")
        )
else:
    # Simple mode: use defaults for race simulation (not shown to user)
    distance_km = 60.0
    stint_km = 2.0
    speed_kmh = 10.0
    switch_time_secs = 40
    fatigue_work_rate = 0.01
    fatigue_tau_recovery = 7.0
    power_speed_exponent = 0.4
    optimize_stint = False
    stint_min, stint_max, stint_step = None, None, None

# --- Solver Settings ---
st.sidebar.markdown("---")
st.sidebar.header(t("solver_settings"))
time_limit = st.sidebar.number_input(
    t("time_limit"),
    min_value=5, max_value=600, value=60, step=5
)
gap_tolerance = st.sidebar.number_input(
    t("gap_tolerance"),
    min_value=0.001, max_value=0.1, value=0.001, step=0.001, format="%.3f"
)

# Main content area
n_paddlers = n_seats + n_resting

# Load defaults from CSV
st.header(t("crew_data"))

# File uploader for custom CSV
uploaded_file = st.file_uploader(
    t("upload_csv"),
    type=['csv'],
    help=t("upload_csv_help")
)

# Load defaults
if uploaded_file is not None:
    default_paddlers, default_ability, default_weight, default_eligibility = load_crew_defaults(
        uploaded_file=uploaded_file
    )
    st.success(t("loaded_paddlers").format(n=len(default_paddlers)))
elif DEFAULT_CSV_PATH.exists():
    default_paddlers, default_ability, default_weight, default_eligibility = load_crew_defaults(
        csv_path=DEFAULT_CSV_PATH
    )
else:
    default_paddlers = None
    default_ability = None
    default_weight = None
    default_eligibility = None

# Paddler editor
st.subheader(t("paddlers"))

if default_paddlers is not None and len(default_paddlers) == n_paddlers:
    # Build editable DataFrame with all paddler data
    paddler_data = pd.DataFrame({
        t('name'): default_paddlers['name'],
        t('ability'): default_ability,
        t('weight_kg'): default_weight
    })
else:
    # Use generic defaults
    paddler_data = pd.DataFrame({
        t('name'): [f"P{i+1}" for i in range(n_paddlers)],
        t('ability'): [1.0] * n_paddlers,
        t('weight_kg'): [75.0] * n_paddlers
    })

edited_paddlers = st.data_editor(
    paddler_data,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={
        t('name'): st.column_config.TextColumn(t('name'), width='medium'),
        t('ability'): st.column_config.NumberColumn(t('ability'), min_value=0.5, max_value=2.0, step=0.01, format="%.2f"),
        t('weight_kg'): st.column_config.NumberColumn(t('weight_kg'), min_value=40, max_value=150, step=1)
    }
)

# Extract paddler data
names = edited_paddlers[t('name')].tolist()
paddler_ability = edited_paddlers[t('ability')].tolist()
paddler_weight = edited_paddlers[t('weight_kg')].tolist()

# Seat Configuration
st.header(t("seat_configuration"))

col1, col2 = st.columns(2)

# Default seat weights from desafio60k2025.qmd
default_seat_weights = [1.05, 1.02, 1.02, 1.01, 1.00, 1.10]
if n_seats != 6:
    default_seat_weights = [1.0] * n_seats

# Default seat entry weights from desafio60k2025.qmd
default_entry_weights = [1.0, 2.0, 1.5, 1.5, 2.0, 1.0]
if n_seats != 6:
    default_entry_weights = [1.0] * n_seats

with col1:
    st.subheader(t("seat_weights"))
    st.markdown(t("seat_weights_desc"))

    seat_weights = []
    cols = st.columns(min(n_seats, 6))
    for i in range(n_seats):
        with cols[i % 6]:
            w = st.number_input(
                f"{t('seat')} {i+1}",
                min_value=0.1, max_value=3.0,
                value=default_seat_weights[i] if i < len(default_seat_weights) else 1.0,
                step=0.01,
                format="%.2f",
                key=f"weight_{i}"
            )
            seat_weights.append(w)

with col2:
    st.subheader(t("seat_entry_weights"))
    st.markdown(t("seat_entry_desc"))

    seat_entry_weights = []
    cols = st.columns(min(n_seats, 6))
    for i in range(n_seats):
        with cols[i % 6]:
            w = st.number_input(
                f"{t('seat')} {i+1}",
                min_value=0.1, max_value=3.0,
                value=default_entry_weights[i] if i < len(default_entry_weights) else 1.0,
                step=0.1,
                key=f"entry_weight_{i}"
            )
            seat_entry_weights.append(w)

# Eligibility Matrix
st.header(t("seat_eligibility"))
st.markdown(t("seat_eligibility_desc"))

# Create eligibility DataFrame for editing
if default_eligibility is not None and default_eligibility.shape == (n_paddlers, n_seats):
    elig_data = {f"{t('seat')} {s+1}": default_eligibility[:, s].astype(bool) for s in range(n_seats)}
else:
    elig_data = {f"{t('seat')} {s+1}": [True] * n_paddlers for s in range(n_seats)}

elig_df = pd.DataFrame(elig_data, index=names)

edited_elig = st.data_editor(
    elig_df,
    use_container_width=True,
    hide_index=False
)

# Convert back to numpy array
eligibility = edited_elig.values.astype(int)

# Manual mode: show schedule editor instead of optimization
if interface_mode == "manual":
    st.header(t("manual_schedule"))
    st.markdown(t("manual_schedule_help"))

    # Determine cycle length (n_paddlers / n_resting = number of stints per cycle)
    cycle_length = n_paddlers // n_resting if n_resting > 0 else n_paddlers

    # Try to get default schedule from last optimization result
    last_result = st.session_state.get('result')
    if last_result and 'cycle_schedule' in last_result:
        last_cycle = last_result['cycle_schedule']
        # Check if dimensions match
        if len(last_cycle) == cycle_length and len(last_cycle.columns) == n_seats:
            # Use last optimization result as default
            schedule_df = last_cycle.copy()
            # Rename columns to current language
            schedule_df.columns = [f"{t('seat')} {s+1}" for s in range(n_seats)]
            schedule_df.index = [f"{t('stint')} {i+1}" for i in range(cycle_length)]
        else:
            # Dimensions don't match, use generic default
            default_schedule = {}
            for s in range(n_seats):
                col_name = f"{t('seat')} {s+1}"
                default_schedule[col_name] = [names[(s + t) % n_paddlers] for t in range(cycle_length)]
            schedule_df = pd.DataFrame(default_schedule)
            schedule_df.index = [f"{t('stint')} {i+1}" for i in range(cycle_length)]
    else:
        # No previous result, use generic default
        default_schedule = {}
        for s in range(n_seats):
            col_name = f"{t('seat')} {s+1}"
            default_schedule[col_name] = [names[(s + t) % n_paddlers] for t in range(cycle_length)]
        schedule_df = pd.DataFrame(default_schedule)
        schedule_df.index = [f"{t('stint')} {i+1}" for i in range(cycle_length)]

    edited_schedule = st.data_editor(
        schedule_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            f"{t('seat')} {s+1}": st.column_config.SelectboxColumn(
                f"{t('seat')} {s+1}",
                options=names,
                required=True
            ) for s in range(n_seats)
        }
    )

    # Analyze button
    if st.button(t("analyze_schedule"), type="primary", use_container_width=True):
        # Validate schedule
        errors = []
        for stint_idx in range(cycle_length):
            stint_paddlers = []
            for seat_idx in range(n_seats):
                paddler_name = edited_schedule.iloc[stint_idx, seat_idx]
                if paddler_name not in names:
                    errors.append(t("invalid_paddler").format(
                        name=paddler_name, stint=stint_idx+1, seat=seat_idx+1))
                if paddler_name in stint_paddlers:
                    errors.append(t("duplicate_paddler").format(
                        name=paddler_name, stint=stint_idx+1))
                stint_paddlers.append(paddler_name)

        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner(t("analyzing")):
                # Create weight dict
                paddler_weights = {name: weight for name, weight in zip(names, paddler_weight)}

                # Analyze
                trim_stats = analyze_manual_schedule(edited_schedule, paddler_weights, n_seats)

                # Store in session state
                st.session_state['manual_trim_stats'] = trim_stats
                st.session_state['manual_schedule'] = edited_schedule
                st.session_state['has_manual_result'] = True

# Optimization mode: show optimize button
else:
    # Clear manual results when not in manual mode
    st.session_state['has_manual_result'] = False

    st.header(t("optimize"))

    if st.button(t("run_optimization"), type="primary", use_container_width=True):
        if len(names) != n_paddlers:
            st.error(t("provide_paddlers_error").format(n=n_paddlers))
        else:
            paddlers = pd.DataFrame({"name": names})

            with st.spinner(t("solving")):
                try:
                    # Check if we should optimize stint length
                    if interface_mode == "full" and optimize_stint and stint_min is not None:
                        # Generate stint range
                        import numpy as np
                        stint_range = tuple(np.arange(stint_min, stint_max + stint_step/2, stint_step))

                        meta_result = optimize_stint_range(
                            paddlers,
                            stint_km_range=stint_range,
                            max_consecutive=max_consecutive,
                            distance_km=distance_km,
                            speed_kmh=speed_kmh,
                            switch_time_secs=switch_time_secs,
                            seat_eligibility=eligibility,
                            seat_weights=seat_weights,
                            seat_entry_weights=seat_entry_weights,
                            paddler_ability=paddler_ability,
                            paddler_weight=paddler_weight,
                            trim_penalty_weight=trim_penalty_weight,
                            moi_penalty_weight=moi_penalty_weight,
                            steerer_paddle_fraction=steerer_paddle_fraction,
                            n_seats=n_seats,
                            n_resting=n_resting,
                            solver_time_secs=time_limit,
                            gap_tolerance=gap_tolerance,
                        )

                        # Use the best result
                        best = meta_result['best']
                        result = solve_rotation_cycle(
                            paddlers,
                            stint_km=best['stint_km'],
                            max_consecutive=max_consecutive,
                            distance_km=distance_km,
                            speed_kmh=speed_kmh,
                            switch_time_secs=switch_time_secs,
                            seat_eligibility=eligibility,
                            seat_weights=seat_weights,
                            seat_entry_weights=seat_entry_weights,
                            paddler_ability=paddler_ability,
                            paddler_weight=paddler_weight,
                            trim_penalty_weight=trim_penalty_weight,
                            moi_penalty_weight=moi_penalty_weight,
                            steerer_paddle_fraction=steerer_paddle_fraction,
                            n_seats=n_seats,
                            n_resting=n_resting,
                            solver_time_secs=time_limit,
                            gap_tolerance=gap_tolerance,
                            fatigue_work_rate=fatigue_work_rate,
                            fatigue_tau_recovery=fatigue_tau_recovery,
                            power_speed_exponent=power_speed_exponent,
                        )

                        st.session_state['meta_result'] = meta_result
                        st.session_state['has_meta_result'] = True
                    else:
                        result = solve_rotation_cycle(
                            paddlers,
                            stint_km=stint_km,
                            max_consecutive=max_consecutive,
                            distance_km=distance_km,
                            speed_kmh=speed_kmh,
                            switch_time_secs=switch_time_secs,
                            seat_eligibility=eligibility,
                            seat_weights=seat_weights,
                            seat_entry_weights=seat_entry_weights,
                            paddler_ability=paddler_ability,
                            paddler_weight=paddler_weight,
                            trim_penalty_weight=trim_penalty_weight,
                            moi_penalty_weight=moi_penalty_weight,
                            steerer_paddle_fraction=steerer_paddle_fraction,
                            n_seats=n_seats,
                            n_resting=n_resting,
                            solver_time_secs=time_limit,
                            gap_tolerance=gap_tolerance,
                            fatigue_work_rate=fatigue_work_rate,
                            fatigue_tau_recovery=fatigue_tau_recovery,
                            power_speed_exponent=power_speed_exponent,
                        )
                        st.session_state['has_meta_result'] = False

                    # Store result in session state
                    st.session_state['result'] = result
                    st.session_state['has_result'] = True
                    st.session_state['result_mode'] = interface_mode

                except Exception as e:
                    st.error(t("optimization_failed").format(error=str(e)))
                    st.session_state['has_result'] = False

# Display manual analysis results
if st.session_state.get('has_manual_result', False):
    st.header(t("manual_analysis_results"))

    trim_stats = st.session_state['manual_trim_stats']
    manual_schedule = st.session_state['manual_schedule']
    cycle_length = len(manual_schedule)

    # Key metrics (same as simple mode)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(t("cycle_length"), f"{cycle_length} {t('stints').lower()}")
    with col2:
        st.metric(t("max_trim"), f"{trim_stats['max_abs_trim_moment']:.1f} kg-m")

    # Generate rotation rules
    cycle_rules = generate_cycle_rules(manual_schedule, names, n_seats)

    # Cycle rules (same as automatic mode)
    st.subheader(t("rotation_rules"))
    st.markdown(t("rotation_rules_desc"))

    rules_df = pd.DataFrame([
        {t('paddler'): name, t('rule'): rule}
        for name, rule in cycle_rules.items()
    ])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    # Colored rotation pattern grid (show 2 cycles like automatic mode)
    st.subheader(t("rotation_pattern"))

    all_paddlers_set = set(names)
    n_rows_to_show = cycle_length * 2  # Show 2 cycles

    st.markdown(t("rotation_pattern_desc").format(n=cycle_length, times="∞"))

    # Build schedule for 2 cycles by repeating
    schedule_2cycles = pd.concat([manual_schedule, manual_schedule], ignore_index=True)
    schedule_2cycles.columns = [f"Banco {s+1}" for s in range(n_seats)]

    # Build matrix with paddlers out (resting)
    out_matrix = []
    for i in range(n_rows_to_show):
        in_canoe = set(schedule_2cycles.iloc[i].values)
        out = sorted(all_paddlers_set - in_canoe)
        out_matrix.append(out)
    out_df = pd.DataFrame(out_matrix, columns=[f"{t('out')} {j+1}" for j in range(n_resting)])

    # Combine in and out
    combined = pd.concat([schedule_2cycles.reset_index(drop=True), out_df], axis=1)
    paddler_to_num = {name: i for i, name in enumerate(names)}
    combined_numeric = combined.replace(paddler_to_num)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(3, n_rows_to_show * 0.6)),
                                    gridspec_kw={'width_ratios': [n_seats, n_resting], 'wspace': 0.05})

    # Left plot: seats (in canoe)
    im1 = ax1.imshow(combined_numeric.iloc[:, :n_seats].values, cmap='tab10', aspect='auto', vmin=0, vmax=9)
    ax1.set_xticks(range(n_seats))
    seat_labels = [f"{t('seat')} {i+1}" for i in range(n_seats)]
    ax1.set_xticklabels(seat_labels, fontsize=9)
    ax1.set_yticks(range(n_rows_to_show))
    ax1.set_yticklabels([f"{t('stint')} {i+1}" for i in range(n_rows_to_show)])

    for i in range(n_rows_to_show):
        for j in range(n_seats):
            name = combined.iloc[i, j]
            ax1.text(j, i, name[:3], ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Highlight first cycle with yellow dashed rectangle
    rect = Rectangle((-0.5, -0.5), n_seats, cycle_length, linewidth=3,
                    edgecolor='yellow', facecolor='none', linestyle='--')
    ax1.add_patch(rect)

    # Right plot: out (resting)
    im2 = ax2.imshow(combined_numeric.iloc[:, n_seats:].values, cmap='tab10', aspect='auto', vmin=0, vmax=9)
    ax2.set_xticks(range(n_resting))
    ax2.set_xticklabels([f"{t('out')} {j+1}" for j in range(n_resting)], fontsize=9)
    ax2.set_yticks(range(n_rows_to_show))
    ax2.set_yticklabels([])

    for i in range(n_rows_to_show):
        for j in range(n_resting):
            name = combined.iloc[i, n_seats + j]
            ax2.text(j, i, name[:3], ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Paddler summary (same as simple mode)
    st.subheader(t("paddler_summary"))
    paddler_summary = generate_paddler_summary(manual_schedule, names, n_seats)
    cols_to_show = ['name', 'stints_paddled', 'stints_rested', 'longest_stretch_stints', 'stints_paddled_per_cycle']
    st.dataframe(paddler_summary[cols_to_show], use_container_width=True, hide_index=True)

    # Balance analysis (trim stats)
    st.subheader(t("balance_analysis"))
    col1, col2 = st.columns(2)
    with col1:
        st.metric(t("max_trim"), f"{trim_stats['max_abs_trim_moment']:.1f} kg-m")
    with col2:
        st.metric(t("avg_moi"), f"{trim_stats['avg_moi']:.1f} kg-m²")

    # Show normalized values for comparison across crews
    if 'normalized_max_abs_trim' in trim_stats:
        st.caption(f"{t('normalized')}: Trim={trim_stats['normalized_max_abs_trim']:.2f}, "
                  f"MOI={trim_stats['normalized_avg_moi']:.2f}")

    # Stint details table
    st.markdown(f"**{t('cycle_stint_details')}**")
    trim_data = pd.DataFrame({
        t('stint'): [f"{t('stint')} {i+1}" for i in range(len(trim_stats['trim_moments']))],
        t('trim_kgm'): [f"{m:+.1f}" for m in trim_stats['trim_moments']],
        t('direction'): [t('stern') if m > 0 else t('bow') if m < 0 else t('neutral')
                      for m in trim_stats['trim_moments']],
        t('moi_kgm2'): [f"{m:.1f}" for m in trim_stats['moi_values']]
    })
    st.dataframe(trim_data, use_container_width=True, hide_index=True)

    # Seat breakdown table showing intermediate calculations
    if 'seat_breakdown' in trim_stats:
        with st.expander(t("seat_breakdown"), expanded=False):
            for stint_idx, stint_seats in enumerate(trim_stats['seat_breakdown']):
                st.markdown(f"**{t('stint')} {stint_idx + 1}**")
                breakdown_rows = []
                for seat_data in stint_seats:
                    breakdown_rows.append({
                        t('seat'): seat_data['seat'],
                        t('paddler'): seat_data['name'],
                        t('weight_kg'): f"{seat_data['weight']:.1f}",
                        t('position_m'): f"{seat_data['position']:+.1f}",
                        t('trim_contrib'): f"{seat_data['trim_contrib']:+.1f}",
                        t('moi_contrib'): f"{seat_data['moi_contrib']:.1f}",
                    })
                # Add total row
                total_trim = sum(s['trim_contrib'] for s in stint_seats)
                total_moi = sum(s['moi_contrib'] for s in stint_seats)
                breakdown_rows.append({
                    t('seat'): '',
                    t('paddler'): f"**{t('total')}**",
                    t('weight_kg'): '',
                    t('position_m'): '',
                    t('trim_contrib'): f"**{total_trim:+.1f}**",
                    t('moi_contrib'): f"**{total_moi:.1f}**",
                })
                breakdown_df = pd.DataFrame(breakdown_rows)
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

# Display optimization results (not shown in manual mode)
if st.session_state.get('has_result', False) and interface_mode != "manual":
    result = st.session_state['result']
    result_mode = st.session_state.get('result_mode', 'full')

    st.header(t("results"))

    # Status and key metrics (mode-dependent)
    if result_mode == "simple":
        # Simple mode: just status and cycle length
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t("status"), result['status'])
        with col2:
            st.metric(t("cycle_length"), f"{result['parameters']['cycle_length']} {t('stints').lower()}")
    else:
        # Full mode: all metrics including race time
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("status"), result['status'])
        with col2:
            race_time = result['race_time']
            st.metric(t("race_time"), f"{race_time:.1f} min ({race_time/60:.2f} h)")
        with col3:
            st.metric(t("avg_output"), f"{result['avg_output']:.1%}")
        with col4:
            st.metric(t("stints"), result['parameters']['n_stints'])

        # Show stint optimization results if available
        if st.session_state.get('has_meta_result', False):
            meta_result = st.session_state['meta_result']
            st.subheader(t("stint_optimization_results"))

            col1, col2 = st.columns([1, 2])
            with col1:
                best_stint = meta_result['best']['stint_km']
                best_time = meta_result['best']['race_time']
                st.metric(t("best_stint"), f"{best_stint:.1f} km")
                st.metric(t("race_time"), f"{best_time:.1f} min")

            with col2:
                st.markdown(f"**{t('stint_comparison')}**")
                summary_df = meta_result['summary'].copy()
                summary_df['race_time'] = summary_df['race_time'].round(1)
                summary_df['avg_output'] = (summary_df['avg_output'] * 100).round(1).astype(str) + '%'
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Cycle rules
    st.subheader(t("rotation_rules"))
    st.markdown(t("rotation_rules_desc"))

    rules_df = pd.DataFrame([
        {t('paddler'): name, t('rule'): rule}
        for name, rule in result['cycle_rules'].items()
    ])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    # Colored rotation pattern grid
    st.subheader(t("rotation_pattern"))

    cycle_length = result['parameters']['cycle_length']
    n_stints_total = result['parameters']['n_stints']
    n_cycles_total = n_stints_total // cycle_length

    # Show 2 cycles or less if race is shorter
    n_rows_to_show = min(cycle_length * 2, n_stints_total)
    n_cycles_shown = n_rows_to_show // cycle_length

    st.markdown(t("rotation_pattern_desc").format(n=cycle_length, times=n_cycles_total))

    schedule_for_grid = result['schedule'].head(n_rows_to_show).copy()
    all_paddlers_set = set(names)

    # Build matrix with paddlers out (resting)
    out_matrix = []
    for i in range(n_rows_to_show):
        in_canoe = set(schedule_for_grid.iloc[i].values)
        out = sorted(all_paddlers_set - in_canoe)
        out_matrix.append(out)
    out_df = pd.DataFrame(out_matrix, columns=[f"{t('out')} {j+1}" for j in range(n_resting)])

    # Combine in and out
    combined = pd.concat([schedule_for_grid.reset_index(drop=True), out_df], axis=1)
    paddler_to_num = {name: i for i, name in enumerate(names)}
    combined_numeric = combined.replace(paddler_to_num)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(3, n_rows_to_show * 0.6)),
                                    gridspec_kw={'width_ratios': [n_seats, n_resting], 'wspace': 0.05})

    # Left plot: seats (in canoe)
    im1 = ax1.imshow(combined_numeric.iloc[:, :n_seats].values, cmap='tab10', aspect='auto', vmin=0, vmax=9)
    ax1.set_xticks(range(n_seats))
    seat_labels = [f"{t('seat')} {i+1}" for i in range(n_seats)]
    ax1.set_xticklabels(seat_labels, fontsize=9)
    ax1.set_yticks(range(n_rows_to_show))
    ax1.set_yticklabels([f"{t('stint')} {i+1}" for i in range(n_rows_to_show)])

    for i in range(n_rows_to_show):
        for j in range(n_seats):
            name = combined.iloc[i, j]
            ax1.text(j, i, name[:3], ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Highlight first cycle with yellow dashed rectangle
    if n_rows_to_show > cycle_length:
        rect = Rectangle((-0.5, -0.5), n_seats, cycle_length, linewidth=3,
                        edgecolor='yellow', facecolor='none', linestyle='--')
        ax1.add_patch(rect)

    # Right plot: out (resting)
    im2 = ax2.imshow(combined_numeric.iloc[:, n_seats:].values, cmap='tab10', aspect='auto', vmin=0, vmax=9)
    ax2.set_xticks(range(n_resting))
    ax2.set_xticklabels([f"{t('out')} {j+1}" for j in range(n_resting)], fontsize=9)
    ax2.set_yticks(range(n_rows_to_show))
    ax2.set_yticklabels([])

    for i in range(n_rows_to_show):
        for j in range(n_resting):
            name = combined.iloc[i, n_seats + j]
            ax2.text(j, i, name[:3], ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Paddler summary (mode-dependent columns)
    st.subheader(t("paddler_summary"))
    paddler_summary = result['paddler_summary'].copy()
    if result_mode == "simple":
        # Simple mode: hide time-related columns
        cols_to_show = ['name', 'stints_paddled', 'stints_rested', 'longest_stretch_stints', 'stints_paddled_per_cycle']
        cols_available = [c for c in cols_to_show if c in paddler_summary.columns]
        st.dataframe(paddler_summary[cols_available], use_container_width=True, hide_index=True)
    else:
        st.dataframe(paddler_summary, use_container_width=True, hide_index=True)

    # Trim stats (always shown for transparency)
    trim_stats = result['parameters'].get('trim_stats')
    if trim_stats:
        st.subheader(t("balance_analysis"))
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t("max_trim"), f"{trim_stats['max_abs_trim_moment']:.1f} kg-m")
        with col2:
            st.metric(t("avg_moi"), f"{trim_stats.get('avg_moi', 0):.1f} kg-m²")

        # Show normalized values for comparison across crews
        if 'normalized_max_abs_trim' in trim_stats:
            st.caption(f"{t('normalized')}: Trim={trim_stats['normalized_max_abs_trim']:.2f}, "
                      f"MOI={trim_stats['normalized_avg_moi']:.2f}")

        st.markdown(f"**{t('cycle_stint_details')}**")
        trim_data = pd.DataFrame({
            t('stint'): [f"{t('stint')} {i+1}" for i in range(len(trim_stats['trim_moments']))],
            t('avg_output'): [f"{o:.1%}" for o in trim_stats.get('cycle_avg_outputs', [0] * len(trim_stats['trim_moments']))],
            t('trim_kgm'): [f"{m:+.1f}" for m in trim_stats['trim_moments']],
            t('direction'): [t('stern') if m > 0 else t('bow') if m < 0 else t('neutral')
                          for m in trim_stats['trim_moments']],
            t('moi_kgm2'): [f"{m:.1f}" for m in trim_stats.get('moi_values', [0] * len(trim_stats['trim_moments']))]
        })
        st.dataframe(trim_data, use_container_width=True, hide_index=True)

        # Seat breakdown table showing intermediate calculations
        if 'seat_breakdown' in trim_stats:
            with st.expander(t("seat_breakdown"), expanded=False):
                for stint_idx, stint_seats in enumerate(trim_stats['seat_breakdown']):
                    st.markdown(f"**{t('stint')} {stint_idx + 1}**")
                    breakdown_rows = []
                    for seat_data in stint_seats:
                        breakdown_rows.append({
                            t('seat'): seat_data['seat'],
                            t('paddler'): seat_data['name'],
                            t('weight_kg'): f"{seat_data['weight']:.1f}",
                            t('position_m'): f"{seat_data['position']:+.1f}",
                            t('trim_contrib'): f"{seat_data['trim_contrib']:+.1f}",
                            t('moi_contrib'): f"{seat_data['moi_contrib']:.1f}",
                        })
                    # Add total row
                    total_trim = sum(s['trim_contrib'] for s in stint_seats)
                    total_moi = sum(s['moi_contrib'] for s in stint_seats)
                    breakdown_rows.append({
                        t('seat'): '',
                        t('paddler'): f"**{t('total')}**",
                        t('weight_kg'): '',
                        t('position_m'): '',
                        t('trim_contrib'): f"**{total_trim:+.1f}**",
                        t('moi_contrib'): f"**{total_moi:.1f}**",
                    })
                    breakdown_df = pd.DataFrame(breakdown_rows)
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # Stats columns (only in full mode)
    if result_mode == "full":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(t("aggregate_stats"))
            stats = result['summary_stats']
            stats_df = pd.DataFrame({
                t('metric'): [
                    t('cycle_length'),
                    t('total_stints'),
                    t('avg_time_per_paddler'),
                    t('max_time_any'),
                    t('min_time_any'),
                    t('max_consecutive_stretch'),
                ],
                t('value'): [
                    f"{stats['cycle_length']} {t('stints').lower()}",
                    f"{stats['n_stints']} {t('stints').lower()}",
                    f"{stats['avg_time_per_paddler_min']:.1f} min",
                    f"{stats['max_time_any_paddler_min']:.1f} min",
                    f"{stats['min_time_any_paddler_min']:.1f} min",
                    f"{stats['max_consecutive_stretch_min']:.1f} min",
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader(t("race_summary"))
            avg_stint_time = result['parameters'].get('avg_stint_time_min', 0)
            summary_df = pd.DataFrame({
                t('metric'): [
                    t('distance'),
                    t('base_speed'),
                    t('stint_distance'),
                    t('avg_stint_time'),
                    t('switch_time'),
                    t('number_of_switches'),
                    t('effective_speed'),
                ],
                t('value'): [
                    f"{distance_km} km",
                    f"{speed_kmh} km/h",
                    f"{stint_km} km",
                    f"{avg_stint_time:.1f} min",
                    f"{switch_time_secs} s",
                    f"{result['parameters']['n_stints'] - 1}",
                    f"{distance_km / (result['race_time'] / 60):.2f} km/h",
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(f"*{t('footer')}*")
