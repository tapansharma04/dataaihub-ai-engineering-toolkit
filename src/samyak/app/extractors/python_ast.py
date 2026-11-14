"""Python AST extractor (stdlib ``ast`` only).

Direct string literals in model-shaped slots become bound model references.
Level 1 same-file string constants may bind a Name to a model id.
Other non-literal slots become AI operations with unresolved or dynamic identity.
No execution, imports, or cross-file lookup.
"""

from __future__ import annotations

import ast

from samyak.app.extractors.constant_bindings import (
    ConstantBindingTable,
    collect_constant_bindings,
)
from samyak.app.models import (
    AiOperationKind,
    ApplicationIdentityState,
    EvidenceType,
    ResolutionMethod,
    SourceLocation,
    UnboundReason,
)
from samyak.app.observations import Observation, ObservationKind, sort_observations

MODEL_SLOT_KEYS = frozenset({"model", "model_id", "model_name"})
MODEL_KWARG_NAMES = frozenset({"model", "model_id"})
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
_MAX_EXPRESSION_CHARS = 160
_GETENV_NAMES = frozenset({"getenv", "getenvb"})


def extract_python_observations(relative_path: str, source: str) -> tuple[Observation, ...]:
    """Parse ``source`` as Python and return model/import observations."""
    tree = ast.parse(source, filename=relative_path)
    bindings = collect_constant_bindings(relative_path, tree)
    visitor = _Extractor(relative_path, bindings)
    visitor.visit(tree)
    return visitor.finish()


class _Extractor(ast.NodeVisitor):
    def __init__(self, relative_path: str, bindings: ConstantBindingTable) -> None:
        self._path = relative_path
        self._bindings = bindings
        self._observations: list[Observation] = []
        self._imported_providers: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            provider = _provider_from_module(alias.name)
            if provider is None:
                continue
            self._imported_providers.add(provider)
            self._observations.append(_import_observation(self._path, node, provider, alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module:
            self.generic_visit(node)
            return
        provider = _provider_from_module(node.module)
        if provider is not None:
            self._imported_providers.add(provider)
            self._observations.append(_import_observation(self._path, node, provider, node.module))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_dict_constructor(node.func):
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                self._emit_dict_slot(keyword.arg, keyword.value)
            self.generic_visit(node)
            return
        symbol = _call_symbol(node.func)
        for keyword in node.keywords:
            if keyword.arg not in MODEL_KWARG_NAMES:
                continue
            self._emit_call_slot(keyword.arg, keyword.value, symbol)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                continue
            key = _string_constant(key_node)
            if key is None:
                continue
            self._emit_dict_slot(key, value_node)
        self.generic_visit(node)

    def _emit_call_slot(self, key: str, value_node: ast.expr, symbol: str | None) -> None:
        literal = _string_constant(value_node)
        if literal is not None:
            self._observations.append(
                Observation(
                    kind=ObservationKind.MODEL_REFERENCE,
                    location=SourceLocation(
                        path=self._path,
                        evidence_type=EvidenceType.PYTHON_KEYWORD,
                        detected_value=literal,
                        line_start=value_node.lineno,
                        line_end=_end_lineno(value_node),
                        symbol=symbol,
                        config_key=key,
                    ),
                )
            )
            return
        if self._emit_resolved_name(
            key, value_node, evidence_type=EvidenceType.PYTHON_KEYWORD, symbol=symbol
        ):
            return
        identity, reason, expression = _unbound_slot(value_node)
        self._observations.append(
            Observation(
                kind=ObservationKind.AI_OPERATION,
                location=SourceLocation(
                    path=self._path,
                    evidence_type=EvidenceType.PYTHON_KEYWORD,
                    detected_value="",
                    line_start=value_node.lineno,
                    line_end=_end_lineno(value_node),
                    symbol=symbol,
                    config_key=key,
                ),
                identity_state=identity,
                unbound_reason=reason,
                expression=expression,
                operation_kind=AiOperationKind.MODEL_INFERENCE,
            )
        )

    def _emit_dict_slot(self, key: str, value_node: ast.expr) -> None:
        if key not in MODEL_SLOT_KEYS:
            return
        literal = _string_constant(value_node)
        if literal is not None:
            self._observations.append(
                Observation(
                    kind=ObservationKind.MODEL_REFERENCE,
                    location=SourceLocation(
                        path=self._path,
                        evidence_type=EvidenceType.PYTHON_LITERAL,
                        detected_value=literal,
                        line_start=value_node.lineno,
                        line_end=_end_lineno(value_node),
                        config_key=key,
                    ),
                )
            )
            return
        if self._emit_resolved_name(
            key, value_node, evidence_type=EvidenceType.PYTHON_LITERAL, symbol=None
        ):
            return
        identity, reason, expression = _unbound_slot(value_node)
        self._observations.append(
            Observation(
                kind=ObservationKind.AI_OPERATION,
                location=SourceLocation(
                    path=self._path,
                    evidence_type=EvidenceType.PYTHON_LITERAL,
                    detected_value="",
                    line_start=value_node.lineno,
                    line_end=_end_lineno(value_node),
                    config_key=key,
                ),
                identity_state=identity,
                unbound_reason=reason,
                expression=expression,
                operation_kind=AiOperationKind.MODEL_INFERENCE,
            )
        )

    def _emit_resolved_name(
        self,
        key: str,
        value_node: ast.expr,
        *,
        evidence_type: EvidenceType,
        symbol: str | None,
    ) -> bool:
        if not isinstance(value_node, ast.Name):
            return False
        binding = self._bindings.resolve(value_node.id)
        if binding is None:
            return False
        self._observations.append(
            Observation(
                kind=ObservationKind.MODEL_REFERENCE,
                location=SourceLocation(
                    path=self._path,
                    evidence_type=evidence_type,
                    detected_value=binding.value,
                    line_start=value_node.lineno,
                    line_end=_end_lineno(value_node),
                    symbol=symbol,
                    config_key=key,
                ),
                expression=value_node.id,
                resolution_method=ResolutionMethod.SAME_FILE_CONSTANT,
                resolution_source=binding.location,
            )
        )
        return True

    def finish(self) -> tuple[Observation, ...]:
        unique_provider: str | None = None
        if len(self._imported_providers) == 1:
            unique_provider = next(iter(self._imported_providers))
        attached: list[Observation] = []
        for item in self._observations:
            if (
                item.kind in {ObservationKind.MODEL_REFERENCE, ObservationKind.AI_OPERATION}
                and item.provider_id is None
                and unique_provider is not None
            ):
                attached.append(
                    Observation(
                        kind=item.kind,
                        location=item.location,
                        provider_id=unique_provider,
                        identity_state=item.identity_state,
                        unbound_reason=item.unbound_reason,
                        expression=item.expression,
                        operation_kind=item.operation_kind,
                        resolution_method=item.resolution_method,
                        resolution_source=item.resolution_source,
                    )
                )
                continue
            attached.append(item)
        return sort_observations(attached)


def _import_observation(path: str, node: ast.stmt, provider: str, detected: str) -> Observation:
    return Observation(
        kind=ObservationKind.PROVIDER_SDK_IMPORT,
        location=SourceLocation(
            path=path,
            evidence_type=EvidenceType.PYTHON_IMPORT,
            detected_value=detected,
            line_start=node.lineno,
            line_end=_end_lineno(node),
            symbol=provider,
        ),
        provider_id=provider,
    )


def _provider_from_module(name: str) -> str | None:
    if name == PROVIDER_OPENAI or name.startswith("openai."):
        return PROVIDER_OPENAI
    if name == PROVIDER_ANTHROPIC or name.startswith("anthropic."):
        return PROVIDER_ANTHROPIC
    return None


def _string_constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _unbound_slot(node: ast.expr) -> tuple[ApplicationIdentityState, UnboundReason, str]:
    identity, reason = _classify_unbound(node)
    return identity, reason, _expression_text(node)


def _classify_unbound(node: ast.expr) -> tuple[ApplicationIdentityState, UnboundReason]:
    if isinstance(node, ast.Name):
        return ApplicationIdentityState.UNRESOLVED, UnboundReason.NAME_NOT_BOUND
    if isinstance(node, ast.Attribute):
        return ApplicationIdentityState.UNRESOLVED, UnboundReason.UNRESOLVED_ATTRIBUTE
    if _is_environment_expression(node):
        return ApplicationIdentityState.DYNAMIC, UnboundReason.DYNAMIC_EXPRESSION
    if isinstance(node, ast.Call):
        return ApplicationIdentityState.DYNAMIC, UnboundReason.DYNAMIC_CALL
    if isinstance(node, ast.Subscript):
        return ApplicationIdentityState.DYNAMIC, UnboundReason.DYNAMIC_SUBSCRIPT
    if isinstance(node, ast.JoinedStr | ast.BinOp | ast.BoolOp | ast.IfExp | ast.Await):
        return ApplicationIdentityState.DYNAMIC, UnboundReason.DYNAMIC_EXPRESSION
    return ApplicationIdentityState.UNRESOLVED, UnboundReason.UNRESOLVED_EXPRESSION


def _is_environment_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in _GETENV_NAMES:
            return True
        if isinstance(func, ast.Attribute) and func.attr in _GETENV_NAMES:
            return True
        return (
            isinstance(func, ast.Attribute) and func.attr == "get" and _is_environ_name(func.value)
        )
    if isinstance(node, ast.Subscript):
        return _is_environ_name(node.value)
    return False


def _is_environ_name(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id == "environ":
        return True
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _expression_text(node: ast.expr) -> str:
    try:
        text = ast.unparse(node)
    except (ValueError, RecursionError):
        text = node.__class__.__name__
    compact = " ".join(text.split())
    if len(compact) <= _MAX_EXPRESSION_CHARS:
        return compact
    return compact[: _MAX_EXPRESSION_CHARS - 3] + "..."


def _end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int):
        return end
    lineno = getattr(node, "lineno", None)
    if isinstance(lineno, int):
        return lineno
    raise TypeError("AST node is missing lineno")


def _call_symbol(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_dict_constructor(func: ast.expr) -> bool:
    return isinstance(func, ast.Name) and func.id == "dict"
