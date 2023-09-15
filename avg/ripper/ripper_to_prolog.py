import re

def extract_ripper_rules(text):
    """
    Extract rules from the given RIPPER output text.
    
    Args:
    - text (str): The RIPPER output text.
    
    Returns:
    - dict: A dictionary with goal states as keys and corresponding rules as values.
    """
    # Split the text by dashed lines to extract sections for each goal state
    sections = re.split(r"-+", text)
    
    rule_dict = {}
    
    # For each section, extract the goal state and corresponding rules
    for section in sections:
        lines = section.strip().split("\n")
        if len(lines) < 2:
            continue
        goal = lines[0].split(":")[0].strip("Accuracy for ")
        rules = re.findall(r"\[.*?\]", lines[1])
        rule_dict[goal] = rules
    
    return rule_dict

def format_goal_name(goal):
    """
    Format the goal name into a proper Prolog predicate name.
    
    Args:
    - goal (str): The original goal name.
    
    Returns:
    - str: The formatted Prolog predicate name.
    """
    goal = goal.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return goal

def ripper_to_prolog(goal, rules):
    """
    Convert the given RIPPER rules to Prolog format with improved parsing.
    
    Args:
    - goal (str): The goal state.
    - rules (list): The RIPPER rules.
    
    Returns:
    - list: A list of Prolog rules.
    """
    prolog_rules = []
    
    for rule in rules:
        # Remove square brackets and split the conjunctions within a rule
        conditions = rule.strip("[]").split('^')
        
        # Convert to Prolog format
        prolog_conditions = ', '.join([f"{cond.split('=')[0]}({cond.split('=')[1]})" for cond in conditions])
        prolog_rule = f"{format_goal_name(goal)} :- {prolog_conditions}."
        prolog_rules.append(prolog_rule)
        
    return prolog_rules

def main():
    # The RIPPER output text
    ripper_output = """
    Accuracy for stopped at Station5: 0.9993
[[ox=j^vx=i^px=g] V [vx=h^ox=j^px=g] V [vy=d^vx=e^ox=j] V [vy=d^vx=j^px=g]]
--------------------------------------------------
Accuracy for moving to Station2: 0.9948
[[ox=g^ow=d^vx=a^wz=g] V [ox=g^ow=d^vx=a^wz=a] V [ox=g^ow=d^vx=a^wz=d] V [ox=g^ow=d^vx=a^wz=f] V [ox=g^ow=d^vx=a^wz=i] V [ox=g^ow=d^vx=a^wz=e] V [ox=g^ow=d^vx=a^wz=b] V [ox=g^ow=d^vx=a] V [ox=g^wz=j^ow=c] V [px=h^ow=b^wz=j^vx=i^ox=j] V [px=h^ow=b^wz=j] V [ox=g^vy=i] V [px=h^vx=i^vy=j^ow=a]]
--------------------------------------------------
Accuracy for moving to Station6: 0.9929
[[vx=a^ow=g] V [ow=e^vx=a] V [ow=h^py=f] V [ow=f^vx=a]]
--------------------------------------------------
Accuracy for initial position: 0.9997
[[vy=e^wz=g^px=g] V [vx=i^py=f^px=g^pz=a^vy=g] V [vy=f^ow=b^px=g^py=f] V [wz=i^vy=c^ox=a] V [vy=d^wz=d^px=g] V [vy=c^ow=b]]
--------------------------------------------------
Accuracy for moving to Station3: 1.0000
[[oy=j^vy=a^ow=a^oz=j] V [vx=j^px=h^vy=g^wz=g] V [vx=j^vy=d^px=h] V [vx=j^vy=c] V [vx=j^vy=g^px=h] V [vx=j^vy=f] V [vx=j^vy=a^ox=j] V [vy=b^ox=j] V [vy=h^oy=j^wz=j^px=h^vx=j] V [vy=e^vx=j]]
--------------------------------------------------
Accuracy for moving to Station5: 0.9990
[[vy=j^px=g^py=e] V [px=f^oz=j] V [vy=h^wz=g] V [vy=i^vx=j^ox=j^px=g] V [vy=h^oy=j^px=g]]
--------------------------------------------------
Accuracy for moving to Station4: 0.9986
[[vy=a^ox=a] V [vy=c^ox=a^ow=c] V [ox=a^oy=j^vx=j] V [vy=c^ox=a^vx=b] V [ox=a^vy=c^vx=e] V [vy=i^ox=a]]
--------------------------------------------------
Accuracy for stopped at Station6: 0.9999
[[px=f^vx=i^py=f] V [px=f^vx=h^py=f] V [vy=c^vx=c^px=f] V [px=f^vy=d] V [vx=d^py=f^px=f]]
--------------------------------------------------
Accuracy for moving to Station1: 0.9971
[[px=h^vx=j^vy=j^oy=j] V [px=h^vx=j^vy=j^wz=a] V [px=h^vx=j^vy=j^wz=g] V [px=h^vx=j^vy=j^wz=d] V [px=h^vy=i^oy=j] V [px=h^ow=b^wz=a] V [px=h^vx=j^vy=j^ow=a^wz=i] V [px=h^vx=j^vy=j^ow=a^wz=f] V [px=h^vx=j^vy=j^ow=a^wz=b] V [px=h^vx=j^vy=j^ow=a^oy=c] V [px=h^vx=j^vy=j^ow=a^wz=h] V [px=h^vx=j^vy=j^ow=a^oy=i] V [px=h^vx=j^vy=j^ow=a^wz=e] V [px=h^ow=c^vx=c] V [px=h^vx=j^oy=a^ow=a]]
--------------------------------------------------
Accuracy for stopped at Station4: 0.9966
[[vx=i^wz=a^ox=a^vy=g] V [vx=i^wz=a^ox=a^vy=e] V [vy=g^wz=a^oz=a^vx=i] V [wz=a^vx=i^ox=a^vy=f] V [vy=g^vx=h^px=g^ow=b] V [wz=a^vy=e^oy=a^px=g] V [vy=g^wz=a^oy=a^px=g^oz=j] V [vx=i^vy=d^ox=a] V [vy=g^px=g^ow=b^py=e] V [wz=a^ow=h^px=g] V [wz=a^ox=e^py=e^px=g] V [wz=a^ox=f^vy=d^px=g] V [vx=h^ox=a^wz=a^ow=a] V [vy=f^wz=b^px=g] V [ox=b^vx=h^px=g] V [vy=h^vx=d]]
--------------------------------------------------
Accuracy for stopped (unknown): 0.9947
[[ox=d^py=e^ow=g^wz=j] V [ox=d^py=e^vx=h] V [vx=i^py=e^oz=a^ow=a^wz=g] V [vx=i^ox=d^vy=d] V [vx=i^py=e^oz=a^wz=f] V [vx=i^ox=d^wz=g] V [oz=a^py=e^vx=i^wz=d] V [oz=a^py=e^ow=g^vy=c] V [oz=a^vy=g^py=e^wz=j^ow=c] V [oz=a^py=e^vy=g^wz=j^ow=d] V [oz=a^vy=g^py=e^wz=j^vx=i^ow=f] V [oz=a^py=e^vy=g^wz=i] V [oz=a^py=e^vy=g^wz=j^ow=a] V [oz=a^py=e^vy=g^wz=j^vx=j] V [oz=a^py=e^vy=g^wz=j^vx=i] V [ox=b^vy=c] V [oz=a^vy=h^py=e] V [oz=a^wz=e^ow=a] V [oz=a^vx=h^py=e^wz=g] V [ox=d^py=e^vy=d] V [vy=c^ox=a^vx=c^ow=e] V [oz=a^wz=i^vx=i] V [vy=c^ox=a^ow=d] V [ox=c^ow=f] V [vy=c^ox=a^ow=a] V [ox=c^vx=e] V [ox=d^py=e^wz=b] V [vx=h^wz=d^ox=a^px=g] V [vx=d^oy=j] V [vx=g^vy=g] V [wz=f^vx=h^px=g^ox=a]]
--------------------------------------------------
Accuracy for stopped at Station1: 0.9956
[[px=h^vx=i^oy=j^vy=g^wz=a] V [ox=j^vx=i^wz=j^vy=g] V [vx=h^px=h^wz=a^ox=j] V [vx=i^ox=j^px=h^vy=f] V [vy=h^vx=i] V [px=h^vx=i^wz=g^oy=j]]
--------------------------------------------------
Accuracy for stopped at Station3: 0.9952
[[vy=d^wz=j^vx=h^px=h] V [px=h^vx=i^vy=d^ox=j] V [vy=g^ox=j^vx=i^wz=j] V [vy=f^wz=j^ox=j^vx=h] V [vy=g^ox=j^oy=a] V [vy=h^vx=i^oy=a]]
--------------------------------------------------
Accuracy for stopped at Station2: 1.0000
[[px=h^ow=g] V [px=h^ow=f] V [px=h^vx=i^ow=e] V [ox=g^vy=h^px=h] V [ox=g^vy=g^px=h] V [px=h^vy=g^oy=a^wz=j] V [ox=g^vx=i]]
--------------------------------------------------
Accuracy for Spawning on floor: 1.0000
[[pz=j]]
--------------------------------------------------

    """
    
    # Extract rules from the RIPPER output
    ripper_rules = extract_ripper_rules(ripper_output)
    
    # Convert RIPPER rules to Prolog format and print
    for goal, rules in ripper_rules.items():
        prolog_rules = ripper_to_prolog(goal, rules)
        for pr in prolog_rules:
            print(pr)
        print("\n")

if __name__ == "__main__":
    main()

