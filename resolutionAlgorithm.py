import sys

def negate_single(c):  # negiraj samo jedan
    return c[1:] if c.startswith('~') else '~' + c

def is_tautology(clause):  # micemo sve tautalogije (a v ~a)
    for lit in clause:
        complement = negate_single(lit)
        if complement in clause:
            return True
    return False

def readFromFile(file_path):  # input
    F = []
    with open(file_path, 'r') as readFile:
        readFile = [line.strip() for line in readFile if not line.strip().startswith("#")]
        for line in readFile:
            line = line.lower()
            literals = [lit.strip() for lit in line.split(" v ")]
            clause = frozenset(literals)
            if not is_tautology(clause):
                F.append(frozenset(literals))
    return F

def negate_clause(G):  # negiramo G
    neG = []
    for lit in G:
        neg_lit = negate_single(lit)
        neG.append(frozenset({neg_lit}))
    return neG

def selectClauses(clauses):  # parove uzimamo
    clauses_list = list(clauses)
    for i in range(len(clauses_list)):
        for j in range(i + 1, len(clauses_list)):
            yield (clauses_list[i], clauses_list[j])

def plResolve(c1, c2):  # gledamo jel mozemo naci nesto
    resolvents = set()
    for lit in c1:
        complement = negate_single(lit)
        if complement in c2:
            new_clause = (c1.union(c2)) - {lit, complement}
            if not is_tautology(new_clause):
                resolvents.add(frozenset(new_clause))
    return resolvents

def plResolution(F, G):
    clause_list = []  # koristimo za pracenje poredka
    parent_list = []
    clause_dict = {}

    for clause in F:
        clause_list.append(clause)
        parent_list.append(0)  # jer ne dolaze od niceg
        clause_dict[clause] = len(clause_list) - 1

    for neg in negate_clause(G):
        if neg not in clause_dict:
            clause_list.append(neg)
            parent_list.append(0)
            clause_dict[neg] = len(clause_list) - 1

    while True:
        new_entries = []
        n = len(clause_list)
        for i in range(n):
            for j in range(i + 1, n):
                resolvents = plResolve(clause_list[i], clause_list[j])
                for r in resolvents:
                    if len(r) == 0:
                        new_entries.append((r, i, j))
                        if r not in clause_dict:
                            clause_list.append(r)
                            parent_list.append((i, j))
                            clause_dict[r] = len(clause_list) - 1
                        return clause_list, parent_list, True
                    if r not in clause_dict:
                        new_entries.append((r, i, j))
        if not new_entries:
            return clause_list, parent_list, False
        else:
            for (r, i, j) in new_entries:
                if r not in clause_dict:
                    clause_list.append(r)
                    parent_list.append((i, j))
                    clause_dict[r] = len(clause_list) - 1

def build_filtered_chain(clause_list, parent_list):
    nil_index = None
    for idx, clause in enumerate(clause_list):  # trazi NIL
        if len(clause) == 0:
            nil_index = idx
            break
    if nil_index is None:
        return [], []

    chain = set()

    def backtrack(i):  # rekurzivno gradi chain
        if i in chain:
            return
        chain.add(i)
        par = parent_list[i]
        if par != 0:
            backtrack(par[0])
            backtrack(par[1])

    backtrack(nil_index)

    filtered_indices = sorted(chain)
    filtered_clauses = [clause_list[i] for i in filtered_indices]

    remap = {old: new for new, old in enumerate(filtered_indices)}
    new_parents = []
    for i in filtered_indices:
        par = parent_list[i]
        if par == 0:
            new_parents.append(0)
        else:
            new_parents.append((remap[par[0]], remap[par[1]]))

    return filtered_clauses, new_parents

def process_command(base, new_clause):
    clause_list, parent_list, success = plResolution(base, frozenset(new_clause))
    goal_str = ' v '.join(new_clause)
    print(f"User's command: {goal_str} ?")
    print("====== Derivation Chain ======")
    filtered_clauses, new_parents = build_filtered_chain(clause_list, parent_list)
    first = True
    for idx, clause in enumerate(filtered_clauses):
        clause_str = ' v '.join(clause) if clause else "NIL"
        par = new_parents[idx]
        if par == 0:
            print(f"{idx}. {clause_str}")
        elif first:
            print("===============")
            print(f"{idx}. {clause_str} ({par[0]}, {par[1]})")
            first = False
        else:
            print(f"{idx}. {clause_str} ({par[0]}, {par[1]})")
    print("===============")
    if success:
        print(f"[CONCLUSION]: {goal_str} is true")
    else:
        print(f"[CONCLUSION]: {goal_str} is unknown")

def process(base, file_path_input):
    with open(file_path_input, 'r') as commands:
        commands = [line.strip() for line in commands if not line.strip().startswith("#")]
        for command in commands:
            if command[-1] == '+':
                new_clause = command[:-1]
                new_clause = frozenset(lit.strip().lower() for lit in new_clause.split('v'))
                base.append(new_clause)
                print(f"User's command: {command}")
                print(f"Added {command[-1]}")
            elif command[-1] == '-':
                removed_clause = command[:-1]
                removed_clause = frozenset(lit.strip().lower() for lit in removed_clause.split('v'))
                print(f"User's command: {command}")
                if removed_clause in base:
                    base.remove(removed_clause)
                    print(f"removed {command[:-1]}")
                else:
                    print(f"{command[:-1]} not found")
            elif command[-1] == '?':
                clause = command[:-1]
                clause = set(lit.strip().lower() for lit in clause.split('v'))
                process_command(base, clause)
            else:
                print("Invalid command:", command)

args = sys.argv
if args[1] == "resolution":
    file_path = args[2]
    F_G = readFromFile(file_path)
    F = F_G[:-1]
    G = F_G[-1]
    clause_list, parent_list, success = plResolution(F, G)
    goal_str = ' v '.join(G)
    print("====== Derivation Chain ======")
    filtered_clauses, new_parents = build_filtered_chain(clause_list, parent_list)
    first = True
    for idx, clause in enumerate(filtered_clauses):
        clause_str = ' v '.join(clause) if clause else "NIL"
        par = new_parents[idx]
        if par == 0:
            print(f"{idx}. {clause_str}")
        elif first:
            print("===============")
            print(f"{idx}. {clause_str} ({par[0]}, {par[1]})")
            first = False
        else:
            print(f"{idx}. {clause_str} ({par[0]}, {par[1]})")
    print("===============")
    if success:
        print(f"[CONCLUSION]: {goal_str} is true")
    else:
        print(f"[CONCLUSION]: {goal_str} is unknown")
elif args[1] == "cooking":
    file_path = args[2]
    file_path_input = args[3]
    F = readFromFile(file_path)
    base = [frozenset(clause) for clause in F]
    print("Constructed with knowledge:")
    for clause in base:
        print(' v '.join(clause))
    print()
    process(base, file_path_input)