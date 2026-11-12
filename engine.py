from knowledge import FAULTS, RULES, QUESTIONS, CONTEXT, get_fault, split_rule

# Maximum number of questions asked in one diagnosis
MAX_Q = 12

# Dangerous symptoms: they are asked first and force at least a CAUTION decision
CRITICAL = {"b_spongy", "b_grind", "e_oillight", "e_knock", "e_gas", "c_milk"}


def context(engine, km, age):
    """Facts taken from the vehicle information form (these are never asked as questions)."""
    facts = set()
    if engine == "Diesel":
        facts.add("fuel_diesel")
    else:
        facts.add("fuel_petrol")          # Petrol and CNG bi-fuel both run on petrol
    if engine.startswith("CNG"):
        facts.add("fuel_cng")
    if km > 5000:
        facts.add("service_overdue")      # derived fact
    if age > 10:
        facts.add("old_vehicle")          # derived fact
    return facts


def is_rule_possible(conditions, answers, ctx):
    """A rule is dead if a context fact is missing or the user already answered 'No' to one symptom."""
    for c in conditions:
        if c in CONTEXT:
            if c not in ctx:
                return False
        elif c in answers and answers[c] is False:
            return False
    return True


def next_question(system, answers, ctx):
    """Adaptive questioning: choose the unanswered symptom that appears in the most promising rules."""
    scores = {}                           # symptom id -> score
    for rule in RULES:
        fault, conditions = split_rule(rule)
        info = get_fault(fault)
        if info["system"] != system or not is_rule_possible(conditions, answers, ctx):
            continue

        # symptoms of this rule that we have not asked yet (context facts are never asked)
        unasked = [c for c in conditions if c not in CONTEXT and c not in answers]
        if not unasked:
            continue

        # a rule with more 'Yes' answers is closer to firing, so it gets a higher weight
        yes_count = sum(1 for c in conditions if answers.get(c) is True)
        weight = 1 + 3 * yes_count
        if info["severity"] == 3:
            weight += 2                   # dangerous faults get extra priority
        for c in unasked:
            scores[c] = scores.get(c, 0) + weight

    if not scores:
        return None                       # nothing left to ask

    for c in CRITICAL:
        if c in scores:
            scores[c] += 5                # safety-critical symptoms are asked first

    # pick the highest score; on a tie, the symptom listed first in knowledge.py wins
    best = None
    for c in QUESTIONS[system]:
        if c in scores and (best is None or scores[c] > scores[best]):
            best = c
    return best


def infer(system, answers, ctx):
    """Forward chaining: check every rule against the known facts and collect the matching faults."""
    yes_answers = [q for q in answers if answers[q] is True]
    facts = set(yes_answers) | set(ctx)   # everything we know to be true
    fired = {}                            # fault -> list of rule numbers that fully matched
    partial = {}                          # fault -> best matching ratio of a half-matched rule

    for number, rule in enumerate(RULES, 1):
        fault, conditions = split_rule(rule)
        if get_fault(fault)["system"] != system:
            continue
        if not is_rule_possible(conditions, answers, ctx):
            continue

        matched = [c for c in conditions if c in facts]
        ratio = len(matched) / len(conditions)
        if ratio == 1:
            fired.setdefault(fault, []).append(number)       # all conditions true: rule fires
        else:
            has_yes = any(c in yes_answers for c in conditions)
            if has_yes and ratio >= 0.5:
                partial[fault] = max(partial.get(fault, 0), ratio)

    results = []
    for fault, rule_numbers in fired.items():
        # base confidence 80%, +7% for every extra rule supporting the same fault, maximum 95%
        confidence = min(95, 80 + 7 * (len(rule_numbers) - 1))
        results.append({"fault": fault, "kind": "Confirmed", "conf": confidence, "rules": rule_numbers})
    for fault, ratio in partial.items():
        if fault not in fired:            # a confirmed fault is never listed as 'Possible'
            results.append({"fault": fault, "kind": "Possible", "conf": int(ratio * 50), "rules": []})

    # Confirmed faults first, then the highest confidence first
    results.sort(key=lambda r: (r["kind"] != "Confirmed", -r["conf"]))
    return results


def conf_label(conf):
    """Convert a confidence percentage into a word."""
    if conf >= 87:
        return "Very High"
    if conf >= 75:
        return "High"
    if conf >= 40:
        return "Moderate"
    return "Low"


def decide(results, answers):
    """Decision making: returns (level, verdict, message). Level 3 = STOP, 2 = CAUTION, 1 = SAFE, 0 = unknown."""
    # level = severity of the most serious confirmed fault
    level = 0
    for r in results:
        if r["kind"] == "Confirmed":
            level = max(level, get_fault(r["fault"])["severity"])

    # safety override: a dangerous symptom means at least CAUTION, even for a minor fault
    critical_hit = any(answers.get(q) is True for q in CRITICAL)
    if critical_hit and level < 2:
        level = 2

    if level == 3:
        return 3, "STOP", "High-risk fault. Do not drive the vehicle; arrange a tow or visit a mechanic immediately."
    if level == 2:
        return 2, "CAUTION", "Drive slowly over short distances and have the vehicle inspected by a mechanic within 1-2 days."
    if level == 1:
        return 1, "SAFE", "Minor issue. Routine maintenance or a DIY fix is sufficient."
    return 0, "INCONCLUSIVE", "No known fault pattern was fully matched. A professional inspection is recommended."