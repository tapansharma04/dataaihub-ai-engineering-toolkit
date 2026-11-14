"""Same-file Level 1 constant bindings.

Not a data-flow engine. A name is resolvable only when the module's
straight-line body has exactly one simple ``Name = "literal"`` (or annotated
equivalent) and the file has no other binding of that name.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from samyak.app.models import EvidenceType, SourceLocation


@dataclass(frozen=True, slots=True)
class ConstantBinding:
    """One qualifying module-level string constant."""

    name: str
    value: str
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class ConstantBindingTable:
    """File-local map of uniquely resolvable names to string constants."""

    _by_name: dict[str, ConstantBinding]

    def resolve(self, name: str) -> ConstantBinding | None:
        return self._by_name.get(name)


def collect_constant_bindings(relative_path: str, tree: ast.AST) -> ConstantBindingTable:
    """Collect L1 bindings from ``tree``. Does not execute source."""
    collector = _Collector(relative_path)
    collector.visit(tree)
    return collector.table()


class _Collector(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self._path = relative_path
        self._depth = 0
        self._qualifying: dict[str, list[ConstantBinding]] = {}
        self._tainted: set[str] = set()

    def table(self) -> ConstantBindingTable:
        resolved: dict[str, ConstantBinding] = {}
        for name, bindings in self._qualifying.items():
            if name in self._tainted or len(bindings) != 1:
                continue
            resolved[name] = bindings[0]
        return ConstantBindingTable(_by_name=resolved)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._taint_args(node.args)
        self._nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._taint_args(node.args)
        self._nested(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._taint_args(node.args)
        self._nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._nested(node)

    def visit_If(self, node: ast.If) -> None:
        self._nested(node)

    def visit_While(self, node: ast.While) -> None:
        self._nested(node)

    def visit_For(self, node: ast.For) -> None:
        self._taint_target(node.target)
        self._nested(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._taint_target(node.target)
        self._nested(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._nested(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._nested(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._taint_target(item.optional_vars)
        self._nested(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._taint_target(item.optional_vars)
        self._nested(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._nested(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._depth == 0 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            literal = _string_constant(node.value)
            if literal is not None:
                self._record(node.targets[0].id, literal, node.targets[0], node.value)
                self.generic_visit(node)
                return
        self._taint_assign_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            self.generic_visit(node)
            return
        if self._depth == 0 and isinstance(node.target, ast.Name):
            literal = _string_constant(node.value)
            if literal is not None:
                self._record(node.target.id, literal, node.target, node.value)
                self.generic_visit(node)
                return
        self._taint_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._taint_target(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._taint_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._taint_target(target)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._tainted.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._tainted.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._tainted.add(alias.asname or alias.name.split(".", 1)[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self._tainted.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._tainted.add(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._taint_target(node.target)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self._tainted.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self._tainted.add(node.name)
        self.generic_visit(node)

    def _nested(self, node: ast.AST) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def _record(self, name: str, value: str, target: ast.AST, value_node: ast.expr) -> None:
        location = SourceLocation(
            path=self._path,
            evidence_type=EvidenceType.PYTHON_CONSTANT,
            detected_value=value,
            line_start=target.lineno,
            line_end=_end_lineno(value_node),
            symbol=name,
        )
        self._qualifying.setdefault(name, []).append(
            ConstantBinding(name=name, value=value, location=location)
        )

    def _taint_assign_targets(self, targets: list[ast.expr]) -> None:
        for target in targets:
            self._taint_target(target)

    def _taint_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._tainted.add(target.id)
            return
        if isinstance(target, ast.Starred):
            self._taint_target(target.value)
            return
        if isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self._taint_target(elt)

    def _taint_args(self, args: ast.arguments) -> None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            self._tainted.add(arg.arg)
        if args.vararg is not None:
            self._tainted.add(args.vararg.arg)
        if args.kwarg is not None:
            self._tainted.add(args.kwarg.arg)


def _string_constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int):
        return end
    lineno = getattr(node, "lineno", None)
    if isinstance(lineno, int):
        return lineno
    raise TypeError("AST node is missing lineno")
