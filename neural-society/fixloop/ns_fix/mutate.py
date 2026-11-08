"""Automatic bug injection (AST mutation operators) for classifier training data.

Each operator produces a mutant source plus a bug class. The hand-written evaluation
tasks (tasks.json) are never generated here; the classifier only ever trains on these.
"""

import ast

CLASSES = ["condition", "arithmetic", "value", "wrong_name", "missing_logic", "syntax"]

# Hand-written task categories -> classifier classes.
TASK_CATEGORY_TO_CLASS = {
    "condition": "condition",
    "arithmetic": "arithmetic",
    "off_by_one": "value",
    "constant": "value",
    "wrong_name": "wrong_name",
    "missing_logic": "missing_logic",
    "syntax": "syntax",
}

_CMP_SWAP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
_CMP_FLIP = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}
_BIN_SWAP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.FloorDiv: ast.Div,
    ast.Div: ast.FloorDiv,
    ast.Mod: ast.FloorDiv,
    ast.Pow: ast.Mult,
}
_ATTR_SWAP = {
    "ljust": "rjust",
    "rjust": "ljust",
    "values": "items",
    "items": "values",
    "keys": "values",
    "lower": "upper",
    "upper": "lower",
    "append": "extend",
    "popleft": "pop",
    "pop": "popleft",
    "startswith": "endswith",
    "endswith": "startswith",
    "strip": "lstrip",
    "lstrip": "rstrip",
    "get": "setdefault",
    "add": "discard",
}
_FUNC_SWAP = {
    "min": "max",
    "max": "min",
    "sorted": "reversed",
    "sum": "max",
    "len": "sum",
    "any": "all",
    "all": "any",
    "abs": "int",
    "range": "enumerate",
}


def _sites(tree):
    """Yield (class, description, apply_fn) for every mutation site."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                for table, tag in ((_CMP_SWAP, "swap"), (_CMP_FLIP, "flip")):
                    if type(op) in table:

                        def f(n=node, i=i, new=table[type(op)]):
                            n.ops[i] = new()

                        yield "condition", f"{tag} comparison", f
        elif isinstance(node, ast.BoolOp):

            def f(n=node):
                n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()

            yield "condition", "swap and/or", f
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):

            def f(n=node):
                n.op = ast.UAdd()  # +x on a bool is truthy-equivalent to x

            yield "condition", "drop not", f
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _BIN_SWAP:

            def f(n=node):
                n.op = _BIN_SWAP[type(n.op)]()

            yield "arithmetic", "swap operator", f
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            for d in (1, -1):

                def f(n=node, d=d):
                    n.value = n.value + d

                yield "value", f"constant {d:+d}", f
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and 0 < len(node.value) < 12:

            def f(n=node):
                n.value = n.value[1:] + n.value[:1] if len(n.value) > 1 else n.value + n.value

            yield "value", "change string", f
        elif isinstance(node, ast.Attribute) and node.attr in _ATTR_SWAP:

            def f(n=node):
                n.attr = _ATTR_SWAP[n.attr]

            yield "wrong_name", "swap method", f
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNC_SWAP:

            def f(n=node):
                n.func.id = _FUNC_SWAP[n.func.id]

            yield "wrong_name", "swap builtin", f
    # wrong variable / misspelled name inside each function
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        names = sorted({n.id for n in ast.walk(fn) if isinstance(n, ast.Name)})
        loads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
        for n in loads:
            for other in names:
                if other != n.id:

                    def f(n=n, other=other):
                        n.id = other

                    yield "wrong_name", "other variable", f

            def f(n=n):
                n.id = n.id[:-1] if len(n.id) > 1 else n.id + "_"

            yield "wrong_name", "misspelled name", f
        # dropped statements and dropped wrappers
        for parent in ast.walk(fn):
            for field in ("body", "orelse"):
                body = getattr(parent, field, None)
                if not isinstance(body, list) or parent is fn and len(body) <= 1:
                    continue
                for i, stmt in enumerate(body):
                    if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)) or (
                        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    ):
                        continue

                    def f(b=body, i=i):
                        b[i] = ast.Pass()

                    yield "missing_logic", f"drop {type(stmt).__name__}", f
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and len(node.args) == 1
                and node.func.id in ("sorted", "max", "min", "abs", "list", "int", "reversed", "set")
            ):
                for parent in ast.walk(fn):
                    for fld, val in ast.iter_fields(parent):
                        if val is node:

                            def f(p=parent, fld=fld, a=node.args[0]):
                                setattr(p, fld, a)

                            yield "missing_logic", "drop wrapper", f
            if isinstance(node, ast.If) and node.orelse == []:

                def f(n=node):
                    n.test = ast.Constant(False)

                yield "missing_logic", "disable branch", f


def _syntax_mutants(src, rng, k):
    lines = src.splitlines(keepends=True)
    out = []
    for _ in range(k * 4):
        i = rng.randrange(len(lines))
        line = lines[i].rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#") or '"""' in line:
            continue
        cands = [j for j, c in enumerate(line) if c in ":)]}("]
        if not cands:
            continue
        j = rng.choice(cands)
        new = lines[:i] + [line[:j] + line[j + 1 :] + "\n"] + lines[i + 1 :]
        text = "".join(new)
        try:
            compile(text, "m.py", "exec")
        except SyntaxError:
            out.append(("syntax", "delete bracket/colon", text))
        if len(out) >= k:
            break
    return out


def mutants(src, rng, per_class=20):
    """Up to `per_class` random mutants of each class: list of (class, description, source)."""
    tree = ast.parse(src)
    by_class = {c: [] for c in CLASSES}
    for idx, (cls, desc, _) in enumerate(_sites(tree)):
        by_class[cls].append((idx, desc))
    out = []
    for cls, sites in by_class.items():
        rng.shuffle(sites)
        for idx, desc in sites[:per_class]:
            t = ast.parse(src)
            for k, (_, _, f) in enumerate(_sites(t)):
                if k == idx:
                    f()
                    break
            try:
                text = ast.unparse(ast.fix_missing_locations(t))
            except Exception:
                continue
            if text != ast.unparse(ast.parse(src)):
                out.append((cls, desc, text))
    out += _syntax_mutants(src, rng, per_class)
    return out
