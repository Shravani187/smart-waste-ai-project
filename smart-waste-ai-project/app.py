"""Unified Streamlit application for the AI-Powered Smart Waste Segregation
and Recycling Management System.

One application, one interface, one navigation menu.
Run with: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np

st.set_page_config(page_title="Smart Waste AI", layout="wide")

PAGES = [
    "Home Dashboard",
    "Waste Segregation & Recycling",
    "Food-Waste Prediction",
    "Practical 2: Prolog Logic",
    "Practical 3: BFS & DFS",
    "Practical 4: A* Search",
    "Practical 5: Minimax Game",
    "Practical 6: Fuzzy Temperature",
    "Practical 7: Washing Machine Fuzzy",
    "Practical 8: ANFIS",
    "Practical 9: Neural Learning",
    "Practical 10: Language Models",
]

st.sidebar.title("Smart Waste AI")
st.sidebar.markdown("**Shravani Kadam | Roll 25**")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Academic project. All data is synthetic.")


def header(title):
    st.title("AI-Powered Smart Waste Segregation & Recycling Management System")
    st.markdown("**Student:** Shravani Kadam | **Roll Number:** 25")
    st.header(title)
    st.markdown("---")


def page_home():
    header("Home Dashboard")
    st.markdown("""
    Welcome to the unified Smart Waste Management System. This single application
    integrates the core waste-management demonstration and all ten AI practicals.
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("AI Practicals", "10")
    c2.metric("Core Modules", "3")
    c3.metric("Data Type", "Synthetic")
    st.subheader("Navigation")
    st.info("Use the sidebar to access any module. Each practical opens inside this same application.")
    st.subheader("Core Waste Management")
    st.markdown("- **Waste Segregation & Recycling** — rule-based item categorisation\n"
                 "- **Food-Waste Prediction** — regression model on synthetic data")
    st.subheader("Ten AI Practicals")
    st.markdown("- Practical 2: Prolog Logic (family + waste rules)\n"
                 "- Practical 3: BFS & DFS pathfinding\n"
                 "- Practical 4: A* informed search\n"
                "- Practical 5: Minimax territory game\n"
                 "- Practical 6: Fuzzy temperature controller\n"
                 "- Practical 7: Mamdani & Sugeno washing machine\n"
                 "- Practical 8: ANFIS student performance\n"
                 "- Practical 9: Supervised + unsupervised neural learning\n"
                 "- Practical 10: Parking-assistant language models")


def page_segregation():
    header("Waste Segregation & Recycling")
    from waste_management.waste_rules import classify_item, needs_collection
    st.info("Rule-based categorisation. NOT image recognition.")
    item = st.text_input("Enter waste item", "paper")
    soiled = st.checkbox("Is the item soiled/contaminated?")
    if st.button("Classify"):
        result = classify_item(item, {"soiled": soiled})
        st.success(f"Category: {result['category']}")
        st.write(f"**Reason:** {result['reason']}")
    st.subheader("Collection Status (illustrative)")
    bins = [("Wet / Biodegradable", 85), ("Dry / Recyclable", 60), ("Residual", 90)]
    for name, level in bins:
        collect = needs_collection("any", level)
        st.write(f"**{name}** bin: {level}% full -> {'Collection needed' if collect else 'OK'}")


def page_food_waste():
    header("Food-Waste Prediction")
    from waste_management.food_waste_prediction import make_synthetic_data, train_model, predict
    st.info("Uses a synthetic dataset. Predictions do not establish real-world accuracy.")
    df = make_synthetic_data()
    model, X_test, y_test, preds, metrics = train_model(df)
    c1, c2, c3, c4 = st.columns(4)
    people = c1.number_input("Number of people", 20, 500, 150)
    meals = c2.number_input("Meals prepared", 1, 4, 3)
    day_type = c3.selectbox("Day type", ["Weekday", "Weekend"])
    hist = c4.number_input("Historical waste (kg)", 0.0, 100.0, 25.0)
    if st.button("Predict Food Waste"):
        dt = 1 if day_type == "Weekend" else 0
        result = predict(model, int(people), int(meals), dt, float(hist))
        st.success(f"Predicted food waste: **{result} kg**")
    st.subheader("Model Metrics")
    st.write(metrics)
    st.subheader("Sample Data")
    st.dataframe(df.head())


def page_prolog():
    header("Practical 2: Prolog Logic")
    st.info("SWI-Prolog required for live execution. Code and queries shown below.")
    st.subheader("Family Relationships & Waste Rules")
    st.markdown("File: `prolog/practical_02_family_and_waste_rules.pl`")
    st.code("""% Sample queries (run in SWI-Prolog):
?- father(john, mary).          % true
?- sibling(mary, tom).          % true
?- grandparent(john, sue).      % true
?- uncle(tom, sue).             % true
?- categorize(plastic_bottle, dry).   % true
?- categorize(newspaper, residual).   % true (soiled paper)
?- needs_collection(wet_bin).        % true""", language="prolog")
    st.subheader("Run Instructions")
    st.code("swipl -s prolog/practical_02_family_and_waste_rules.pl", language="bash")
    try:
        import subprocess
        r = subprocess.run(["swipl", "-s", "prolog/practical_02_family_and_waste_rules.pl",
                            "-g", "father(john, mary),halt."], capture_output=True, text=True, timeout=5)
        if "true" in r.stdout.lower():
            st.success(f"SWI-Prolog available. Query result: {r.stdout.strip()}")
        else:
            st.warning("SWI-Prolog not detected. The .pl file is provided for manual execution.")
    except Exception:
        st.warning("SWI-Prolog not detected. The .pl file is provided for manual execution.")


def page_bfs_dfs():
    header("Practical 3: BFS & DFS Rescue Robot")
    from practical_03_bfs_dfs import run as run_search, render, GRID
    (bfs_path, bfs_steps), (dfs_path, dfs_steps) = run_search()
    c1, c2 = st.columns(2)
    c1.subheader("BFS (shortest path)")
    c1.code(render(GRID, bfs_path))
    c1.write(f"Steps: {bfs_steps}")
    c2.subheader("DFS (not guaranteed shortest)")
    c2.code(render(GRID, dfs_path))
    c2.write(f"Steps: {dfs_steps}")
    st.info("BFS finds the shortest path in an unweighted graph because it explores "
            "level by level. DFS may find a longer path first.")


def page_astar():
    header("Practical 4: A* Search")
    from practical_04_astar import run as run_astar, render, GRID, START, TARGET
    (path, cost), (dpath, dcost) = run_astar()
    st.subheader("A* Result")
    st.code(render(GRID, path))
    st.write(f"Optimal path cost: **{cost}**")
    st.subheader("Dijkstra Baseline")
    st.write(f"Dijkstra cost: **{dcost}**")
    st.info("A* uses Manhattan distance heuristic f(n)=g(n)+h(n). "
            "With h=0 it reduces to Dijkstra. Both find the optimal cost here.")


def page_minimax():
    header("Practical 5: Minimax Territory Control Game")
    from practical_05_minimax import run as run_game
    st.info("AI (X) plays against a greedy opponent (O) on a 5x5 board. Depth limit: 4 plies.")
    if st.button("Run Game"):
        final = run_game(verbose=False)
        from practical_05_minimax import render
        st.code(render(final))
        x = sum(row.count("X") for row in final)
        o = sum(row.count("O") for row in final)
        st.write(f"Final score -> X (AI): {x}, O (opponent): {o}")
        st.success(f"Winner: {'X (AI)' if x > o else 'O' if o > x else 'Draw'}")


def page_fuzzy_temp():
    header("Practical 6: Fuzzy Logic Temperature Controller")
    from practical_06_temperature_fuzzy import controller_output
    st.info("Standalone simulation. Not connected to real equipment.")
    c1, c2 = st.columns(2)
    error = c1.slider("Temperature error (setpoint - actual)", -10.0, 10.0, 2.0)
    delta = c2.slider("Change in error", -3.0, 3.0, 0.0)
    output = controller_output(error, delta)
    st.metric("Controller Output", f"{output:+.3f}")
    st.caption("Positive = heat, negative = cool, zero = hold")
    st.info("Membership functions and simulation plots saved as PNG files when run from CLI.")


def page_washing():
    header("Practical 7: Mamdani & Sugeno Washing Machine")
    from practical_07_washing_machine_fuzzy import mamdani, sugeno
    st.info("Determines water level from load size and dirt level.")
    c1, c2 = st.columns(2)
    load = c1.slider("Load size (kg)", 0, 12, 5)
    dirt = c2.slider("Dirt level (0-100)", 0, 100, 50)
    m = mamdani(load, dirt)
    s = sugeno(load, dirt)
    st.metric("Mamdani Water Level", f"{m:.2f} L")
    st.metric("Sugeno Water Level", f"{s:.2f} L")
    st.caption("Mamdani uses output aggregation + defuzzification. "
               "Sugeno uses weighted average of crisp consequents.")


def page_anfis():
    header("Practical 8: ANFIS Student Performance")
    from practical_08_anfis import ANFIS, make_data
    st.info("Hybrid neuro-fuzzy system. Synthetic data.")
    X, y = make_data()
    split = int(0.8 * len(X))
    model = ANFIS(X[:split]).fit(X[:split], y[:split], epochs=100, lr=0.05)
    pred = model.predict(X[split:])
    c1, c2 = st.columns(2)
    c1.metric("Test MAE", f"{np.mean(np.abs(pred - y[split:])):.2f}")
    c2.metric("Test R2", f"{1 - np.sum((y[split:]-pred)**2)/np.sum((y[split:]-y[split:].mean())**2):.3f}")
    st.subheader("Sample Predictions")
    for i in range(min(5, len(pred))):
        st.write(f"  Hours={X[split+i,0]:.1f}, Attendance={X[split+i,1]:.0f} "
                 f"-> pred={pred[i]:.1f}, actual={y[split+i]:.1f}")


def page_neural():
    header("Practical 9: Supervised & Unsupervised Neural Learning")
    from practical_09_neural_learning import make_parking_data, make_daily_profiles, SOM
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    st.subheader("Part A: MLP Parking Occupancy Prediction")
    X, y = make_parking_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    mlp.fit(X_tr, y_tr)
    preds = mlp.predict(X_te)
    st.write(f"MAE: {mean_absolute_error(y_te, preds):.2f}")
    st.write(f"R2: {r2_score(y_te, preds):.3f}")
    st.subheader("Part B: SOM Parking-Area Grouping")
    profiles = make_daily_profiles()
    som = SOM(n_nodes=3, seed=42).fit(profiles)
    labels = som.predict(profiles)
    st.write("Assigned groups:")
    st.write({f"Area {i}": int(labels[i]) for i in range(len(labels))})


def page_language():
    header("Practical 10: Micro & Small Language Models")
    from practical_10_language_models import BigramLM, NeuralSLM, detect_intent, fill_response, PARKING_TABLE, CORPUS
    st.info("All parking data is synthetic/illustrative, not live.")
    bigram = BigramLM(CORPUS)
    slm = NeuralSLM(CORPUS)
    c1, c2 = st.columns(2)
    c1.subheader("Micro LM (Bigram)")
    seed = c1.text_input("Seed word", "there")
    if c1.button("Generate (Bigram)"):
        c1.write(bigram.generate(seed))
    c2.subheader("Neural SLM")
    seed2 = c2.text_input("Seed words (comma-separated)", "there,are")
    if c2.button("Generate (SLM)"):
        c2.write(slm.generate([w.strip() for w in seed2.split(",")]))
    st.subheader("Intent Detection")
    query = st.text_input("Ask about parking", "How many spaces are available?")
    if st.button("Detect Intent & Respond"):
        intent = detect_intent(query)
        area = list(PARKING_TABLE.keys())[0]
        st.success(f"Intent: {intent}")
        st.write(fill_response(intent, area))


PAGE_MAP = {
    "Home Dashboard": page_home,
    "Waste Segregation & Recycling": page_segregation,
    "Food-Waste Prediction": page_food_waste,
    "Practical 2: Prolog Logic": page_prolog,
    "Practical 3: BFS & DFS": page_bfs_dfs,
    "Practical 4: A* Search": page_astar,
    "Practical 5: Minimax Game": page_minimax,
    "Practical 6: Fuzzy Temperature": page_fuzzy_temp,
    "Practical 7: Washing Machine Fuzzy": page_washing,
    "Practical 8: ANFIS": page_anfis,
    "Practical 9: Neural Learning": page_neural,
    "Practical 10: Language Models": page_language,
}

PAGE_MAP[page]()
