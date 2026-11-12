import base64
from datetime import date

import pandas as pd
import streamlit as st

from knowledge import (SYSTEMS, QUESTIONS, QLABEL, CONTEXT, FAULTS, RULES, MAKES, MODELS,
                       get_fault, split_rule)
from engine import MAX_Q, context, next_question, infer, decide, conf_label

st.set_page_config(page_title="AutoDoc | Vehicle Fault Diagnosis", page_icon="🚗", layout="centered")

# ------------------------------------------------------------------
# Constants: car picture, styling and labels
# ------------------------------------------------------------------
# The car is an SVG drawing. We encode it in base64 so it can be shown as an <img> tag.
CAR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 260">
<defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#2563eb"/><stop offset="1" stop-color="#1e3a8a"/></linearGradient></defs>
<ellipse cx="320" cy="238" rx="280" ry="12" fill="#0f172a" opacity=".15"/>
<path d="M60 180 Q60 150 100 142 L190 130 Q230 78 300 74 L400 74 Q450 78 490 128 L555 140 Q590 148 590 178 L590 196 L60 196 Z" fill="url(#g)"/>
<path d="M215 128 Q245 92 300 90 L342 90 L342 128 Z" fill="#bfdbfe"/>
<path d="M352 90 L398 90 Q430 94 458 128 L352 128 Z" fill="#bfdbfe"/>
<rect x="560" y="150" width="28" height="14" rx="6" fill="#fde68a"/>
<rect x="60" y="152" width="18" height="12" rx="5" fill="#ef4444"/>
<circle cx="170" cy="198" r="38" fill="#111827"/><circle cx="170" cy="198" r="20" fill="#9ca3af"/>
<circle cx="470" cy="198" r="38" fill="#111827"/><circle cx="470" cy="198" r="20" fill="#9ca3af"/>
</svg>"""
CAR_B64 = base64.b64encode(CAR_SVG.encode()).decode()

# CSS: this only changes how the page looks, it has no effect on the logic
STYLE = """<style>
.block-container{padding-top:1.5rem;max-width:860px}
.hero{background:linear-gradient(135deg,#0f172a,#1e3a8a);padding:1.6rem 2rem;border-radius:16px;margin-bottom:.5rem}
.hero h1{color:#fff!important;margin:0;font-size:2.2rem}
.hero p{color:#cbd5e1;margin:.3rem 0 0}
.car{width:100%;max-width:520px;display:block;margin:0 auto}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(15,23,42,.08);color:#0f172a;margin-bottom:.6rem}
.card.q{font-size:1.25rem;font-weight:600;padding:1.6rem 1.4rem}
.muted{color:#64748b;font-size:.85rem}
.badge{color:#fff;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:600}
.stepper{display:flex;margin:0 0 1.2rem 0}
.step{flex:1;text-align:center;font-size:.72rem;color:#94a3b8}
.step span{display:block;width:28px;height:28px;line-height:28px;border-radius:50%;margin:0 auto 4px;background:#e2e8f0;color:#475569;font-weight:700}
.step.cur{color:#1e3a8a;font-weight:700}
.step.cur span{background:#1e3a8a;color:#fff}
.step.done span{background:#16a34a;color:#fff}
[data-testid="stSidebar"] .stButton button{width:100%}
</style>"""
st.markdown(STYLE, unsafe_allow_html=True)

SEVERITY_NAME = {1: "Low", 2: "Medium", 3: "High"}
SEVERITY_COLOR = {1: "#16a34a", 2: "#d97706", 3: "#dc2626"}
STEP_NAMES = ["Home", "Vehicle", "Problem", "Diagnosis", "Results", "Repair"]
PAGE_NUMBER = {"home": 0, "vehicle": 1, "system": 2, "diag": 3, "results": 4, "repair": 5}
ENGINES = ["Petrol", "Diesel", "CNG (Bi-fuel)"]
TOTAL_QUESTIONS = sum(len(q) for q in QUESTIONS.values())

# ------------------------------------------------------------------
# Session state: values that Streamlit remembers between button clicks
# ------------------------------------------------------------------
ss = st.session_state
ss.setdefault("page", "home")                    # which page is currently shown
ss.setdefault("vehicle", {"make": "Suzuki", "model": "Alto", "year": 2018,
                          "engine": "Petrol", "km": 3000, "budget": 20000})
ss.setdefault("system", None)                    # selected system code (E, B, L, T, C)
ss.setdefault("history", [])                     # list of (question id, True/False) answers
ss.setdefault("cache", {})                       # remembers answers so Back can restore them


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def go(page):
    """Switch to another page."""
    ss.page = page


def new_diagnosis():
    """Clear the previous diagnosis and go to Problem Selection."""
    ss.history = []
    ss.cache = {}
    ss.system = None
    ss.page = "system"


def back_to_diag():
    """Go back from Results to the last question (undo the last answer)."""
    if ss.history:
        ss.history.pop()
    ss.page = "diag"


def severity_badge(severity):
    """Small coloured label such as 'High severity'."""
    return (f'<span class="badge" style="background:{SEVERITY_COLOR[severity]}">'
            f'{SEVERITY_NAME[severity]} severity</span>')


def show_verdict(level, title, message):
    """Show the decision in a red, orange, green or blue box."""
    text = f"**Decision: {title}**  \n{message}"
    if level == 3:
        st.error(text)
    elif level == 2:
        st.warning(text)
    elif level == 1:
        st.success(text)
    else:
        st.info(text)


def budget_check(info, budget):
    """Compare the repair cost with the user's budget."""
    if info["cost_high"] <= budget:
        return "Within"
    if info["cost_low"] <= budget:
        return "Possibly"
    return "Over budget"


def condition_text(condition):
    """Readable text of a rule condition (a question or a vehicle fact)."""
    if condition in CONTEXT:
        return CONTEXT[condition]
    return QLABEL[condition]


def vehicle_facts():
    """Facts derived from the vehicle form: fuel type, service overdue, old vehicle."""
    v = ss.vehicle
    age = date.today().year - v["year"]
    return context(v["engine"], v["km"], age)


def current_results():
    """Run the inference engine on the answers given so far."""
    if not ss.system:
        return {}, []
    answers = {}
    for question, answer in ss.history:
        answers[question] = answer
    return answers, infer(ss.system, answers, vehicle_facts())


def stepper(page):
    """Progress bar at the top: finished steps are green, the current step is blue."""
    current = PAGE_NUMBER[page]
    html = '<div class="stepper">'
    for i, name in enumerate(STEP_NAMES):
        if i < current:
            css = "done"
        elif i == current:
            css = "cur"
        else:
            css = ""
        html += f'<div class="step {css}"><span>{i + 1}</span>{name}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def build_dot(hit_faults):
    """Full decision tree in Graphviz DOT language: root -> systems -> faults. Diagnosed path is red."""
    lines = ['digraph G {rankdir=LR; node [shape=box, style="rounded,filled", fillcolor=white, fontsize=10];',
             'root [label="Vehicle Problem", shape=oval];']
    hit_systems = set()
    for key in hit_faults:
        hit_systems.add(get_fault(key)["system"])

    for code, (name, icon, desc) in SYSTEMS.items():
        color = "red" if code in hit_systems else "gray"
        lines.append(f'{code} [label="{name}", color={color}];')
        lines.append(f'root -> {code} [color={color}];')

    for key in FAULTS:
        info = get_fault(key)
        if key in hit_faults:
            color, fill = "red", "mistyrose"
        else:
            color, fill = "gray", "white"
        lines.append(f'{key} [label="{info["name"]}", color={color}, fillcolor={fill}];')
        lines.append(f'{info["system"]} -> {key} [color={color}];')
    return "\n".join(lines) + "}"


def path_dot():
    """Decision tree of the user's own answers: question -> Yes/No -> next question -> ... -> fault."""
    def clean(text):
        return text.replace('"', "'")

    code = ss.system
    _, results = current_results()
    lines = ['digraph{rankdir=TB; node[shape=box,style=rounded,fontsize=10];',
             f'S[label="{SYSTEMS[code][0]} system",shape=oval];']
    previous, edge_label = "S", ""
    for i, (question, answer) in enumerate(ss.history):
        lines.append(f'N{i}[label="{clean(QUESTIONS[code][question])}"];')
        if edge_label:
            lines.append(f'{previous}->N{i}[label="{edge_label}"];')
        else:
            lines.append(f'{previous}->N{i};')
        previous = f"N{i}"
        edge_label = "Yes" if answer else "No"

    if results:
        end_text = get_fault(results[0]["fault"])["name"]
    else:
        end_text = "No definite fault"
    lines.append(f'R[label="{clean(end_text)}",color=red];')
    lines.append(f'{previous}->R[label="{edge_label}"];')
    return "\n".join(lines) + "}"


# Two small sample decision trees (fixed diagrams for explanation)
START_TREE = '''digraph{rankdir=TB; node[shape=box,style=rounded,fontsize=10];
q1[label="What happens when you turn the key?"];
q1->q2[label="Nothing / clicking"]; q1->q3[label="Engine cranks"];
q2[label="Headlights or horn weak?"];
q2->dead[label="Yes"]; dead[label="Weak / Dead Battery",color=red];
q2->q4[label="No"]; q4[label="Clicking sound?"];
q4->starter[label="Yes"]; starter[label="Starter / Relay Fault",color=red];
q4->chk[label="No"]; chk[label="Check terminals, fuses, wiring",color=red];
q3[label="Engine stalls immediately?"];
q3->pump[label="Yes"]; pump[label="Fuel Pump Failure",color=red];
q3->q5[label="No"]; q5[label="Diesel engine in cold weather?"];
q5->glow[label="Yes"]; glow[label="Glow Plug Failure",color=red];
q5->scan[label="No"]; scan[label="OBD scan: sensors / injectors",color=red];}'''

HEAT_TREE = '''digraph{rankdir=TB; node[shape=box,style=rounded,fontsize=10];
q1[label="Engine overheating?"];
q1->q2[label="Yes"]; q2[label="Coolant level dropping / leak?"];
q2->q3[label="Yes"]; q3[label="White smoke from exhaust?"];
q3->hg[label="Yes"]; hg[label="Head Gasket Failure (STOP)",color=red];
q3->cl[label="No"]; cl[label="Coolant Leak",color=red];
q2->q4[label="No"]; q4[label="Heater not blowing hot air?"];
q4->th[label="Yes"]; th[label="Stuck Thermostat",color=red];
q4->wp[label="No"]; wp[label="Check radiator fan and water pump",color=red];}'''


def build_report(candidates, title, message):
    """Text report that the user can download."""
    v = ss.vehicle
    lines = ["AutoDoc Diagnosis Report", "=" * 25,
             f"Vehicle: {v['year']} {v['make']} {v['model']} ({v['engine']})",
             f"System: {SYSTEMS[ss.system][0]}",
             f"Decision: {title} - {message}", ""]
    for r in candidates:
        info = get_fault(r["fault"])
        lines.append(f"{info['name']} - {r['kind']} ({r['conf']}%), {SEVERITY_NAME[info['severity']]} severity, "
                     f"estimated cost Rs {info['cost_low']:,}-{info['cost_high']:,}")
        lines.append("Possible causes: " + info["causes"])
        for i, step in enumerate(info["steps"], 1):
            lines.append(f"  {i}. {step}")
        lines.append("Precaution: " + info["precaution"])
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Page 1: Home Dashboard
# ------------------------------------------------------------------
def page_home():
    stepper("home")
    st.markdown('<div class="hero"><h1>AutoDoc</h1>'
                '<p>Vehicle Fault Diagnosis &amp; Repair Decision Support System</p></div>',
                unsafe_allow_html=True)
    st.markdown(f'<img class="car" src="data:image/svg+xml;base64,{CAR_B64}">', unsafe_allow_html=True)

    st.markdown("#### System Overview")
    st.write("AutoDoc is a rule-based expert system that diagnoses faults in five vehicle systems. "
             "It asks adaptive yes/no questions, applies forward-chaining inference over its knowledge base, "
             "and produces a safety-aware decision with repair guidance for Petrol, Diesel and CNG vehicles.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicle systems", len(SYSTEMS))
    c2.metric("Knowledge rules", len(RULES))
    c3.metric("Diagnostic questions", TOTAL_QUESTIONS)
    c4.metric("Known faults", len(FAULTS))

    st.markdown("#### How it works")
    steps = [("1. Describe", "Enter your vehicle details and choose the problem area."),
             ("2. Answer", "Respond to a short series of adaptive yes/no questions."),
             ("3. Act", "Get the identified fault, severity, cost and repair steps.")]
    for column, (title, text) in zip(st.columns(3), steps):
        column.markdown(f'<div class="card"><b>{title}</b><br><span class="muted">{text}</span></div>',
                        unsafe_allow_html=True)
    st.button("Start Diagnosis", type="primary", on_click=go, args=("vehicle",))


# ------------------------------------------------------------------
# Page 2: Vehicle Information
# ------------------------------------------------------------------
def page_vehicle():
    stepper("vehicle")
    st.subheader("Vehicle Information")
    v = ss.vehicle
    c1, c2 = st.columns(2)
    make = c1.selectbox("Make", MAKES, index=MAKES.index(v["make"]))
    models = MODELS[make]                      # the model list depends on the selected make
    model_index = models.index(v["model"]) if v["model"] in models else 0
    model = c2.selectbox("Model", models, index=model_index)
    year = c1.number_input("Model year", 1990, date.today().year, v["year"])
    engine = c2.radio("Engine type", ENGINES, index=ENGINES.index(v["engine"]), horizontal=True)
    km = c1.number_input("Kilometres since last service", 0, 100000, v["km"], step=500)
    budget = c2.number_input("Repair budget (Rs)", 0, 1000000, v["budget"], step=1000)

    back_col, next_col = st.columns(2)
    if back_col.button("Back"):
        ss.page = "home"
        st.rerun()
    if next_col.button("Continue", type="primary"):
        ss.vehicle = {"make": make, "model": model, "year": int(year), "engine": engine,
                      "km": int(km), "budget": int(budget)}
        ss.page = "system"
        st.rerun()


# ------------------------------------------------------------------
# Page 3: Problem Selection (radio buttons: click the circle to select)
# ------------------------------------------------------------------
def system_option_text(code):
    """Text shown next to each radio circle: icon, system name and short description."""
    name, icon, desc = SYSTEMS[code]
    return f"{icon}  **{name}**  -  {desc}"


def page_system():
    stepper("system")
    st.subheader("Problem Selection")
    st.write("Which system is giving you trouble? Select one option.")

    codes = list(SYSTEMS.keys())
    if ss.system in codes:
        start_index = codes.index(ss.system)   # remember the earlier choice
    else:
        start_index = None                      # nothing selected at first
    choice = st.radio("Problem area", codes, index=start_index,
                      format_func=system_option_text, label_visibility="collapsed")

    back_col, next_col = st.columns(2)
    if back_col.button("Back"):
        ss.page = "vehicle"
        st.rerun()
    if next_col.button("Start Diagnosis", type="primary", disabled=choice is None):
        ss.system = choice
        ss.history = []
        ss.cache = {}
        ss.page = "diag"
        st.rerun()


# ------------------------------------------------------------------
# Page 4: Interactive Diagnosis (one question at a time)
# ------------------------------------------------------------------
def page_diag():
    stepper("diag")
    code = ss.system
    answers = {}
    for question, answer in ss.history:
        answers[question] = answer

    # ask the engine for the next question; stop after MAX_Q questions or when nothing is left
    if len(ss.history) < MAX_Q:
        q = next_question(code, answers, vehicle_facts())
    else:
        q = None
    if q is None:
        ss.page = "results"
        st.rerun()

    number = len(ss.history) + 1
    st.subheader(f"{SYSTEMS[code][0]} Diagnosis")
    st.progress((number - 1) / MAX_Q, text=f"Question {number} (maximum {MAX_Q})")
    st.markdown(f'<div class="card q">{QUESTIONS[code][q]}</div>', unsafe_allow_html=True)

    # if the user came back to this question, show the answer given earlier
    earlier = ss.cache.get(q)
    if earlier is None:
        start_index = None
    elif earlier:
        start_index = 0
    else:
        start_index = 1
    choice = st.radio("Answer", ["Yes", "No"], index=start_index, horizontal=True,
                      key=f"r_{q}_{number}", label_visibility="collapsed")

    back_col, next_col = st.columns(2)
    if back_col.button("Back"):
        if ss.history:
            ss.history.pop()                    # undo the last answer
        else:
            ss.page = "system"
        st.rerun()
    if next_col.button("Next", type="primary", disabled=choice is None):
        ss.history.append((q, choice == "Yes"))
        ss.cache[q] = (choice == "Yes")
        st.rerun()

    if ss.history:
        with st.expander("Your answers so far"):
            for question, answer in ss.history:
                st.write(f"**{'Yes' if answer else 'No'}** - {QLABEL[question]}")


# ------------------------------------------------------------------
# Page 5: Diagnosis Results
# ------------------------------------------------------------------
def page_results():
    stepper("results")
    answers, results = current_results()
    level, title, message = decide(results, answers)
    st.subheader("Diagnosis Results")
    show_verdict(level, title, message)

    if results:
        top = results[0]                        # the best match is always first
        info = get_fault(top["fault"])
        if top["kind"] == "Confirmed":
            label = "Identified fault"
        else:
            label = "Most likely fault (not confirmed)"
        st.markdown(f'<div class="card"><div class="muted">{label}</div>'
                    f'<h3 style="margin:.2rem 0">{info["name"]}</h3>{severity_badge(info["severity"])}</div>',
                    unsafe_allow_html=True)
        st.write(f"**Confidence level:** {top['conf']}% ({conf_label(top['conf'])})")
        st.progress(top["conf"] / 100)
        st.markdown("**Possible causes**")
        st.write(info["causes"])

        if len(results) > 1:
            st.markdown("**Other possible faults**")
            rows = []
            for r in results[1:5]:
                other = get_fault(r["fault"])
                rows.append({"Fault": other["name"], "Status": r["kind"], "Confidence (%)": r["conf"],
                             "Severity": SEVERITY_NAME[other["severity"]]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        if top["rules"]:
            with st.expander("Why this diagnosis?"):
                for number in top["rules"]:
                    _, conditions = split_rule(RULES[number - 1])
                    parts = []
                    for c in conditions:
                        if c in CONTEXT:
                            parts.append(CONTEXT[c])
                        else:
                            parts.append("Yes to: " + QLABEL[c])
                    st.caption(f"Rule R{number}: " + "  AND  ".join(parts) + f"  ->  {info['name']}")
    else:
        st.info("None of the known fault patterns matched your answers.")

    b1, b2, b3 = st.columns(3)
    b1.button("Back", on_click=back_to_diag)
    b2.button("Repair Recommendations", type="primary", on_click=go, args=("repair",), disabled=not results)
    b3.button("New Diagnosis", on_click=new_diagnosis)


# ------------------------------------------------------------------
# Page 6: Repair Recommendations
# ------------------------------------------------------------------
def page_repair():
    stepper("repair")
    answers, results = current_results()
    level, title, message = decide(results, answers)

    confirmed = [r for r in results if r["kind"] == "Confirmed"]
    if confirmed:
        candidates = confirmed[:3]              # confirmed faults come first
    else:
        candidates = results[:3]                # otherwise show the possible faults

    st.subheader("Repair Recommendations")
    budget = ss.vehicle["budget"]
    tab_names = [get_fault(r["fault"])["name"] for r in candidates]
    for tab, r in zip(st.tabs(tab_names), candidates):
        info = get_fault(r["fault"])
        with tab:
            st.markdown(severity_badge(info["severity"]), unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated cost", f"Rs {info['cost_low']:,} - {info['cost_high']:,}")
            m2.metric("Repair type", "DIY possible" if info["diy"] else "Mechanic required")
            m3.metric("Budget check", budget_check(info, budget))
            st.markdown("#### Step-by-step solution")
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(info["steps"], 1)))
            st.markdown("#### Precautions")
            st.warning(info["precaution"])

    st.caption("Costs are approximate Pakistani market estimates and vary by city and workshop.")
    b1, b2, b3 = st.columns(3)
    b1.button("Back to Results", on_click=go, args=("results",))
    b2.download_button("Download Report", build_report(candidates, title, message), "autodoc_report.txt")
    b3.button("New Diagnosis", on_click=new_diagnosis)


# ------------------------------------------------------------------
# Sidebar pages: Decision Tree and Knowledge Base
# ------------------------------------------------------------------
def page_tree():
    st.subheader("Decision Trees")
    _, results = current_results()
    hit_faults = set()
    for r in results:
        if r["kind"] == "Confirmed":
            hit_faults.add(r["fault"])

    tab1, tab2, tab3, tab4 = st.tabs(["Diagnosis Tree", "Your Decision Path",
                                      "Sample: Won't Start", "Sample: Overheating"])
    with tab1:
        st.caption("All systems and faults. Your diagnosed path is highlighted in red.")
        st.graphviz_chart(build_dot(hit_faults))
    with tab2:
        if ss.system and ss.history:
            st.graphviz_chart(path_dot())
        else:
            st.info("Complete a diagnosis to see the decision path of your answers.")
    with tab3:
        st.graphviz_chart(START_TREE)
    with tab4:
        st.graphviz_chart(HEAT_TREE)
    st.button("Back to Home", on_click=go, args=("home",))


def page_kb():
    st.subheader("Knowledge Base")
    st.write(f"**{len(RULES)}** rules | **{len(FAULTS)}** faults | **{TOTAL_QUESTIONS}** diagnostic questions")
    rows = []
    for number, rule in enumerate(RULES, 1):
        fault, conditions = split_rule(rule)
        info = get_fault(fault)
        rows.append({"Rule": f"R{number}",
                     "System": SYSTEMS[info["system"]][0],
                     "IF": " AND ".join(condition_text(c) for c in conditions),
                     "THEN": info["name"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.button("Back to Home", on_click=go, args=("home",))


# ------------------------------------------------------------------
# Sidebar and page routing (decides which page function runs)
# ------------------------------------------------------------------
with st.sidebar:
    st.title("AutoDoc")
    st.button("Home", on_click=go, args=("home",))
    st.button("Decision Tree", on_click=go, args=("tree",))
    st.button("Knowledge Base", on_click=go, args=("kb",))
    st.caption("Rule-based expert system with forward chaining, decision making and decision support.")

if ss.page == "home":
    page_home()
elif ss.page == "vehicle":
    page_vehicle()
elif ss.page == "system":
    page_system()
elif ss.page == "diag":
    page_diag()
elif ss.page == "results":
    page_results()
elif ss.page == "repair":
    page_repair()
elif ss.page == "tree":
    page_tree()
elif ss.page == "kb":
    page_kb()