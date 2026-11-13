# AutoDoc: Vehicle Fault Diagnosis Expert System

Rule-based expert system (Python + Streamlit) for Petrol, Diesel and CNG vehicles.

- knowledge.py : knowledge base (systems, questions, faults, rules)
- engine.py    : inference engine (forward chaining, adaptive questions, decision making)
- app.py       : Streamlit user interface (6 pages + decision tree + knowledge base)

Run locally:  pip install -r requirements.txt  then  streamlit run app.py

Live Deployed Application (Streamlit): https://autodoc-expert-system.streamlit.app/
