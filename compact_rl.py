"""CRL (Compact RiverLang) transpiler.

Converts between standard RiverLang and a compact token-efficient format
designed for faster LLM agent response times.

Two main functions:
  - riverlang_to_compact(rl_text) -> compact CRL string
  - compact_to_modelset(crl_text) -> ModelSet ready for Compiler
"""

from __future__ import annotations

import re
from typing import Any

from riverlang.ast.parser import RiverLangParser
from riverlang.ast.riverlang_ast import (
    Element,
    Entity,
    EntityGenerator,
    EntitySetGenerator,
    Expression,
    FileSource,
    Grouping,
    GroupFieldExpression,
    GroupFieldIdentifier,
    IdentifierExpression,
    Model,
    ModelSet,
    OrderDirection,
    OrderedExpression,
    Property,
    Relation,
    RelationType,
    SchemaField,
    SchemaInfo,
    Source,
    Split,
    TableSource,
    Type,
    Window,
    Constant,
)

# ---------------------------------------------------------------------------
# Type abbreviation maps
# ---------------------------------------------------------------------------

_TYPE_TO_CRL = {"string": "str", "timestamp": "ts", "timestamptz": "tsz", "duration": "dur"}
_CRL_TO_TYPE = {v: k for k, v in _TYPE_TO_CRL.items()}
_ANNOTATION_TO_CRL = {"dimension": "dim", "metric": "met"}
_CRL_TO_ANNOTATION = {v: k for k, v in _ANNOTATION_TO_CRL.items()}


def _type_to_crl(ty: Type) -> str:
    """Convert a Type AST node to compact type string."""
    grammar = ty.to_grammar()
    return _TYPE_TO_CRL.get(grammar, grammar)


def _crl_to_type(s: str) -> Type:
    """Parse a compact type string into a Type AST node."""
    s = s.strip()
    expanded = _CRL_TO_TYPE.get(s, s)
    # Handle composed types like list[str], option[int], etc.
    m = re.match(r"^(\w+)\[(.+)\]$", expanded)
    if m:
        from riverlang.ast.riverlang_ast import ComposeType
        container = m.group(1)
        inner = _crl_to_type(m.group(2))
        return Type(ty=ComposeType(container=container, element=inner))
    # Handle parameterized types like decimal(18,2)
    m = re.match(r"^(\w+)\((.+)\)$", expanded)
    if m:
        from riverlang.ast.riverlang_ast import IntParamType
        base = m.group(1)
        params = [int(p.strip()) for p in m.group(2).split(",")]
        return Type(ty=IntParamType(base_type=base, int_params=params))
    return Type(ty=expanded)


def _id_to_display(id_str: str) -> str:
    """Convert snake_case id to display name: customer_id -> Customer Id."""
    return id_str.replace("_", " ").title()


def _annotation_to_crl(ann: str) -> str:
    """Convert annotation to compact form: dimension -> dim."""
    return _ANNOTATION_TO_CRL.get(ann, ann)


def _crl_to_annotation(ann: str) -> str:
    """Convert compact annotation back: dim -> dimension."""
    return _CRL_TO_ANNOTATION.get(ann, ann)


# ---------------------------------------------------------------------------
# RL -> CRL  (AST walk -> compact text)
# ---------------------------------------------------------------------------


def riverlang_to_compact(rl_text: str) -> str:
    """Convert standard RiverLang text to CRL compact format."""
    ms = RiverLangParser.from_str(rl_text)
    lines: list[str] = []
    for model in ms.models:
        lines.append(f"model {model.id} {{")
        for elem in model.elements:
            el = elem.element
            if isinstance(el, Source):
                lines.extend(_emit_source(elem))
            elif isinstance(el, Entity):
                lines.extend(_emit_entity(elem))
            elif isinstance(el, Constant):
                lines.extend(_emit_constant(elem))
        lines.append("}")
    return "\n".join(lines)


def _emit_source(elem: Element) -> list[str]:
    src: Source = elem.element
    if isinstance(src.source, TableSource):
        header = f'src {elem.id} = table("{src.source.table_id}") {{'
    else:
        header = f'src {elem.id} = file("{src.source.path_or_url}", "{src.source.format}") {{'
    lines = [header]
    # Schema fields
    for field in src.schema_info.fields:
        lines.append(f"  {field.id}:{_type_to_crl(field.type)}")
    # Windows
    for win in src.windows:
        lines.append(f"  {_emit_window(win)}")
    # Relations
    for rel in src.relations:
        lines.append(f"  {_emit_relation(rel)}")
    lines.append("}")
    return lines


def _emit_entity(elem: Element) -> list[str]:
    ent: Entity = elem.element
    gen = ent.generator

    lines_prefix = []
    # Emit entity description as CRL comment if present
    if elem.description:
        lines_prefix.append(f"# {elem.description}")

    parts = [f"ent {elem.id}"]

    if isinstance(gen, EntityGenerator):
        base_str = ".".join(gen.base.value)
        parts.append(f"from {base_str}")
        if gen.filter:
            parts.append(f"filter {gen.filter.to_grammar()}")
        if gen.grouping:
            parts.append(gen.grouping.to_grammar())
        if gen.split:
            parts.append(gen.split.to_grammar())
        if gen.drop_duplicates:
            parts.append("dedup")
        if gen.order_by:
            order_exprs = ", ".join(e.to_grammar() for e in gen.order_by)
            parts.append(f"order by {order_exprs}")
        if gen.limit is not None:
            parts.append(f"limit {gen.limit}")
        if gen.offset is not None:
            parts.append(f"offset {gen.offset}")
    elif isinstance(gen, EntitySetGenerator):
        op_name = gen.operation.value
        entities = ", ".join(".".join(e.value) for e in gen.entities)
        parts.append(f"from set {op_name}({entities})")

    if ent.is_output:
        parts.append("output")
    parts.append("{")

    header = " ".join(parts)
    lines = lines_prefix + [header]

    # Windows
    for win in ent.windows:
        lines.append(f"  {_emit_window(win)}")
    # Properties
    for prop in ent.properties:
        prop_str = _emit_property(prop)
        # If property has a comment prefix, split into separate lines
        if prop_str.startswith("# "):
            comment_end = prop_str.index("\n") if "\n" in prop_str else len(prop_str)
            # Find where the actual property definition starts (after comment lines)
            prop_lines = prop_str.split("\n")
            for pl in prop_lines:
                lines.append(f"  {pl}")
        else:
            lines.append(f"  {prop_str}")
    # Relations
    for rel in ent.relations:
        lines.append(f"  {_emit_relation(rel)}")
    lines.append("}")
    return lines


def _emit_property(prop: Property) -> str:
    sf = prop.schema_field
    parts: list[str] = []
    # Add description as comment line if non-trivial
    comment = ""
    desc = sf.description.strip() if sf.description else ""
    if desc and desc != "" and desc != sf.display_name:
        comment = f"# {desc}\n"
    for ann in sf.annotations:
        parts.append(f"@{_annotation_to_crl(ann)}")
    type_str = _type_to_crl(sf.type) if sf.type else "auto"
    expr_str = prop.expression.to_grammar()
    # Use .X shorthand for simple base.X passthrough
    if (isinstance(prop.expression.exp, IdentifierExpression)
            and len(prop.expression.exp.value) == 2
            and prop.expression.exp.value[0] == "base"):
        expr_str = f".{prop.expression.exp.value[1]}"
    parts.append(f"{sf.id}:{type_str} = {expr_str}")
    return comment + " ".join(parts)


def _emit_relation(rel: Relation) -> str:
    target_str = ".".join(rel.target.value)
    card = rel.relation_type.value
    parts = [f"rel {rel.id} -> {card} {target_str}"]
    if rel.using:
        parts.append(f"on {rel.using.to_grammar()}")
    return " ".join(parts)


def _emit_window(win: Window) -> str:
    parts = [f"win {win.id}"]
    if win.partition_by:
        pexprs = ", ".join(e.to_grammar() for e in win.partition_by)
        parts.append(f"partition {pexprs}")
    if win.order_by:
        oexprs = ", ".join(e.to_grammar() for e in win.order_by)
        parts.append(f"order {oexprs}")
    return " ".join(parts)


def _emit_constant(elem: Element) -> list[str]:
    const: Constant = elem.element
    type_str = _type_to_crl(const.type)
    expr_str = const.value.to_grammar()
    return [f"const {elem.id}:{type_str} = {expr_str}"]


# ---------------------------------------------------------------------------
# CRL -> AST  (line-based parser -> ModelSet)
# ---------------------------------------------------------------------------

class CRLParseError(Exception):
    """Error parsing CRL text."""
    def __init__(self, message: str, line_num: int = 0, line: str = ""):
        self.line_num = line_num
        self.line_text = line
        super().__init__(f"Line {line_num}: {message}\n  {line}")


def compact_to_modelset(crl_text: str) -> ModelSet:
    """Parse CRL compact text and return a ModelSet ready for the Compiler."""
    lines = crl_text.split("\n")
    models: list[Model] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("model "):
            model, i = _parse_model(lines, i)
            models.append(model)
        else:
            i += 1
    return ModelSet(models=models)


def _parse_model(lines: list[str], start: int) -> tuple[Model, int]:
    line = lines[start].strip()
    # model <id> {
    m = re.match(r"^model\s+(\w+)\s*\{", line)
    if not m:
        raise CRLParseError("Expected 'model <id> {'", start + 1, line)
    model_id = m.group(1)
    elements: list[Element] = []
    i = start + 1
    while i < len(lines):
        line = lines[i].strip()
        if line == "}" or line.startswith("}"):
            i += 1
            break
        if line.startswith("src "):
            elem, i = _parse_source(lines, i, model_id)
            elements.append(elem)
        elif line.startswith("ent "):
            elem, i = _parse_entity(lines, i, model_id)
            elements.append(elem)
        elif line.startswith("const "):
            elem, i = _parse_constant(lines, i)
            elements.append(elem)
        elif not line or line.startswith("#"):
            i += 1
        else:
            raise CRLParseError(f"Unexpected line in model", i + 1, line)
    model = Model(
        id=model_id,
        name=_id_to_display(model_id),
        description="",
        elements=elements,
    )
    return model, i


def _parse_source(lines: list[str], start: int, model_id: str) -> tuple[Element, int]:
    line = lines[start].strip()
    # src <id> = table("<name>") {  OR  src <id> = file("<path>", "<fmt>") {
    m = re.match(r'^src\s+(\w+)\s*=\s*table\("([^"]+)"\)\s*\{', line)
    if m:
        src_id = m.group(1)
        table_source = TableSource(table_id=m.group(2))
        source_obj: TableSource | FileSource = table_source
    else:
        m = re.match(r'^src\s+(\w+)\s*=\s*file\("([^"]+)",\s*"([^"]+)"\)\s*\{', line)
        if not m:
            raise CRLParseError("Expected src definition", start + 1, line)
        src_id = m.group(1)
        source_obj = FileSource(path_or_url=m.group(2), format=m.group(3))

    fields: list[SchemaField] = []
    relations: list[Relation] = []
    windows: list[Window] = []
    i = start + 1
    while i < len(lines):
        inner = lines[i].strip()
        if inner == "}" or inner.startswith("}"):
            i += 1
            break
        if inner.startswith("rel "):
            relations.append(_parse_relation_line(inner))
            i += 1
        elif inner.startswith("win "):
            windows.append(_parse_window_line(inner))
            i += 1
        elif not inner or inner.startswith("#"):
            i += 1
        else:
            # Source field: id:type
            for field_str in _split_semicolons(inner):
                fields.append(_parse_source_field(field_str.strip()))
            i += 1

    elem = Element(
        id=src_id,
        display_name=_id_to_display(src_id),
        description="",
        element=Source(
            source=source_obj,
            schema_info=SchemaInfo(fields=fields),
            relations=relations,
            windows=windows,
        ),
    )
    return elem, i


def _parse_entity(lines: list[str], start: int, model_id: str) -> tuple[Element, int]:
    # Collect header — may span multiple lines until we find the opening {
    header_line = lines[start].strip()
    i = start + 1

    # If header doesn't end with {, gather continuation lines
    while not header_line.rstrip().endswith("{") and i < len(lines):
        next_line = lines[i].strip()
        if not next_line:
            i += 1
            continue
        header_line = header_line + " " + next_line
        i += 1

    # Parse entity header: ent <id> from <base> [clauses...] [output] {
    m = re.match(r"^ent\s+(\w+)\s+(.+)$", header_line)
    if not m:
        raise CRLParseError("Expected ent definition", start + 1, header_line)
    ent_id = m.group(1)
    rest = m.group(2)

    # Check for output flag and extract the { at end
    is_output = False
    if not rest.rstrip().endswith("{"):
        raise CRLParseError("Expected '{' at end of ent header", start + 1, header_line)
    rest = rest.rstrip()[:-1].strip()  # Remove trailing {

    if rest.endswith("output"):
        is_output = True
        rest = rest[:-6].strip()

    # Parse generator clause
    generator = _parse_generator_clause(rest)

    # Parse body: properties, relations, windows (i already past header)
    properties: list[Property] = []
    relations: list[Relation] = []
    windows: list[Window] = []
    while i < len(lines):
        inner = lines[i].strip()
        if inner == "}" or (inner.startswith("}") and not inner.startswith("}}")):
            i += 1
            break
        if inner.startswith("rel "):
            relations.append(_parse_relation_line(inner))
            i += 1
        elif inner.startswith("win "):
            windows.append(_parse_window_line(inner))
            i += 1
        elif not inner or inner.startswith("#"):
            i += 1
        else:
            # Could be multi-line expression or annotation-on-separate-line
            expr_lines = [inner]
            i += 1
            # If line is ONLY annotations (e.g. "@dim" or "@met @key"), join with next line
            if re.match(r"^(@\w+\s*)+$", inner):
                while i < len(lines):
                    next_stripped = lines[i].strip()
                    if next_stripped and not next_stripped.startswith("}"):
                        expr_lines.append(next_stripped)
                        i += 1
                        if re.match(r"^(@\w+\s*)+$", next_stripped):
                            continue  # More annotations, keep going
                        break  # Got the property line
                    else:
                        break
            else:
                # Gather continuation lines for multi-line expressions
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    if (next_line and next_line[0] in (" ", "\t")
                            and next_stripped
                            and not next_stripped.startswith("rel ")
                            and not next_stripped.startswith("win ")
                            and not next_stripped.startswith("@")
                            and not re.match(r"^\w+:", next_stripped)
                            and next_stripped != "}"):
                        expr_lines.append(next_stripped)
                        i += 1
                    else:
                        break
            full_line = " ".join(expr_lines)
            properties.append(_parse_property_line(full_line))

    entity = Entity(
        properties=properties,
        relations=relations,
        windows=windows,
        generator=generator,
        is_output=is_output,
    )
    elem = Element(
        id=ent_id,
        display_name=_id_to_display(ent_id),
        description="",
        element=entity,
    )
    return elem, i


def _parse_generator_clause(clause: str) -> EntityGenerator | EntitySetGenerator:
    """Parse the generator part of an entity header."""
    clause = clause.strip()

    # Set generator: from set UnionAll(ent1, ent2)
    m = re.match(r"^from\s+set\s+(\w+)\((.+)\)$", clause)
    if m:
        from riverlang.ast.riverlang_ast import EntitySetOps
        op = EntitySetOps(m.group(1))
        entities = [
            IdentifierExpression(value=e.strip().split("."))
            for e in m.group(2).split(",")
        ]
        return EntitySetGenerator(operation=op, entities=entities)

    # Entity generator: from <base> [filter ...] [group ... as ...] [split ... as ...] [dedup] [order by ...] [limit N] [offset M]
    if not clause.startswith("from "):
        raise CRLParseError(f"Expected 'from' in generator clause: {clause}")
    clause = clause[5:].strip()

    # Use a regex-based approach to parse the clause sequentially
    base_str, rest = _extract_until_keyword(clause, ["filter", "group", "split", "dedup", "order", "limit", "offset"])
    base = IdentifierExpression(value=base_str.strip().split("."))

    filter_expr = None
    grouping = None
    split = None
    drop_duplicates = False
    order_by = None
    limit = None
    offset = None

    while rest:
        rest = rest.strip()
        if rest.startswith("filter "):
            rest = rest[7:]
            filter_str, rest = _extract_until_keyword(rest, ["group", "split", "dedup", "order", "limit", "offset"])
            filter_str = _expand_dot_shorthand(filter_str.strip())
            filter_str = _sanitize_expression(filter_str)
            filter_expr = RiverLangParser.parse_expression(filter_str)
        elif rest.startswith("group "):
            rest = rest[6:]
            group_str, rest = _extract_until_keyword(rest, ["split", "dedup", "order", "limit", "offset"])
            grouping = _parse_grouping(group_str.strip())
        elif rest.startswith("split "):
            rest = rest[6:]
            split_str, rest = _extract_until_keyword(rest, ["dedup", "order", "limit", "offset"])
            split = _parse_split(split_str.strip())
        elif rest.startswith("dedup"):
            drop_duplicates = True
            rest = rest[5:]
        elif rest.startswith("order by "):
            rest = rest[9:]
            order_str, rest = _extract_until_keyword(rest, ["limit", "offset"])
            order_by = _parse_order_by(order_str.strip())
        elif rest.startswith("order "):
            # Allow "order" without "by" as shorthand
            rest = rest[6:]
            order_str, rest = _extract_until_keyword(rest, ["limit", "offset"])
            order_by = _parse_order_by(order_str.strip())
        elif rest.startswith("limit "):
            rest = rest[6:]
            limit_str, rest = _extract_until_keyword(rest, ["offset"])
            limit = int(limit_str.strip())
        elif rest.startswith("offset "):
            rest = rest[7:]
            offset = int(rest.strip())
            rest = ""
        else:
            break

    return EntityGenerator(
        base=base,
        filter=filter_expr,
        grouping=grouping,
        split=split,
        drop_duplicates=drop_duplicates,
        order_by=order_by,
        limit=limit,
        offset=offset,
    )


def _extract_until_keyword(text: str, keywords: list[str]) -> tuple[str, str]:
    """Extract text until the next keyword, respecting parentheses and quotes."""
    depth = 0
    in_string = False
    string_char = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == string_char and (i == 0 or text[i - 1] != "\\"):
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            i += 1
            continue
        if ch in ("(", "{", "["):
            depth += 1
            i += 1
            continue
        if ch in (")", "}", "]"):
            depth -= 1
            i += 1
            continue
        if depth == 0:
            for kw in keywords:
                if text[i:].startswith(kw + " ") or text[i:] == kw:
                    return text[:i].rstrip(), text[i:]
        i += 1
    return text, ""


def _parse_grouping(group_str: str) -> Grouping:
    """Parse group clause: 'field1, field2 as grp' or '(expr) as alias, field as grp'."""
    # The last 'as NAME' is the group name
    # Find the last ' as ' that's not inside parens
    parts = _split_last_as(group_str)
    fields_str = parts[0].strip()
    group_name = parts[1].strip()

    group_fields = []
    for field_str in _split_group_fields(fields_str):
        field_str = field_str.strip()
        if not field_str:
            continue
        # Check for (expr) as alias pattern
        m = re.match(r"^\((.+)\)\s+as\s+(\w+)$", field_str)
        if m:
            expr_str = _expand_dot_shorthand(m.group(1).strip())
            expr = RiverLangParser.parse_expression(expr_str)
            group_fields.append(GroupFieldExpression(expression=expr, alias=m.group(2)))
        else:
            # Simple identifier
            ident = field_str.strip().split(".")
            group_fields.append(GroupFieldIdentifier(identifier=ident))

    return Grouping(group_properties=group_fields, group_name=group_name)


def _split_last_as(s: str) -> tuple[str, str]:
    """Find the last ' as ' at depth 0 and split there."""
    depth = 0
    last_as = -1
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in ("(", "{", "["):
            depth += 1
        elif ch in (")", "}", "]"):
            depth -= 1
        elif depth == 0 and s[i:i + 4] == " as " and i + 4 <= len(s):
            last_as = i
        i += 1
    if last_as == -1:
        raise CRLParseError(f"No 'as' found in group clause: {s}")
    return s[:last_as], s[last_as + 4:]


def _split_group_fields(s: str) -> list[str]:
    """Split group fields by comma, respecting parentheses."""
    result = []
    depth = 0
    current = []
    for ch in s:
        if ch in ("(", "{", "["):
            depth += 1
            current.append(ch)
        elif ch in (")", "}", "]"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current))
    return result


def _parse_split(split_str: str) -> Split:
    """Parse split clause: 'expr as alias'."""
    m = re.match(r"^(.+)\s+as\s+(\w+)$", split_str)
    if not m:
        raise CRLParseError(f"Expected 'expr as alias' in split: {split_str}")
    expr_str = _expand_dot_shorthand(m.group(1).strip())
    return Split(
        expression=RiverLangParser.parse_expression(expr_str),
        alias=m.group(2),
    )


def _parse_order_by(order_str: str) -> list[OrderedExpression]:
    """Parse order by clause."""
    result = []
    for part in _split_group_fields(order_str):
        part = part.strip()
        if not part:
            continue
        direction = OrderDirection.ASC
        if part.endswith(" desc"):
            direction = OrderDirection.DESC
            part = part[:-5].strip()
        elif part.endswith(" asc"):
            part = part[:-4].strip()
        expr_str = _expand_dot_shorthand(part)
        result.append(OrderedExpression(
            expression=RiverLangParser.parse_expression(expr_str),
            direction=direction,
        ))
    return result


def _parse_constant(lines: list[str], start: int) -> tuple[Element, int]:
    line = lines[start].strip()
    m = re.match(r"^const\s+(\w+):(\S+)\s*=\s*(.+)$", line)
    if not m:
        raise CRLParseError("Expected const definition", start + 1, line)
    const_id = m.group(1)
    type_obj = _crl_to_type(m.group(2))
    expr = RiverLangParser.parse_expression(m.group(3).strip())
    elem = Element(
        id=const_id,
        display_name=_id_to_display(const_id),
        description="",
        element=Constant(value=expr, type=type_obj),
    )
    return elem, start + 1


def _parse_source_field(field_str: str) -> SchemaField:
    """Parse a source field: 'id:type'."""
    m = re.match(r"^(\w+):(.+)$", field_str)
    if not m:
        raise CRLParseError(f"Expected 'id:type' source field: {field_str}")
    fid = m.group(1)
    return SchemaField(
        id=fid,
        display_name=_id_to_display(fid),
        description="",
        type=_crl_to_type(m.group(2).strip()),
    )


def _parse_property_line(line: str) -> Property:
    """Parse entity property: '[@ann] id:type = expr'."""
    line = line.strip()
    annotations: list[str] = []
    # Extract leading annotations
    while line.startswith("@"):
        m = re.match(r"^@(\w+)\s+", line)
        if not m:
            break
        annotations.append(_crl_to_annotation(m.group(1)))
        line = line[m.end():]

    # Parse id:type = expr
    m = re.match(r"^(\w+):(\S+)\s*=\s*(.+)$", line, re.DOTALL)
    if not m:
        raise CRLParseError(f"Expected 'id:type = expr' property: {line}")
    prop_id = m.group(1)
    type_str = m.group(2)
    expr_str = m.group(3).strip()

    # Expand .X shorthand to base.X
    expr_str = _expand_dot_shorthand(expr_str)

    # Sanitize common LLM expression mistakes
    expr_str = _sanitize_expression(expr_str)

    expr = RiverLangParser.parse_expression(expr_str)

    return Property(
        schema_field=SchemaField(
            id=prop_id,
            display_name=_id_to_display(prop_id),
            description="",
            type=_crl_to_type(type_str),
            annotations=annotations,
        ),
        expression=expr,
    )


def _parse_relation_line(line: str) -> Relation:
    """Parse relation: 'rel id -> one|many target [on expr]'."""
    m = re.match(r"^rel\s+(\w+)\s*->\s*(\w+)\s+(\S+)(?:\s+on\s+(.+))?$", line)
    if not m:
        raise CRLParseError(f"Expected relation definition: {line}")
    rel_id = m.group(1)
    cardinality = m.group(2)
    target_str = m.group(3)
    using_str = m.group(4)

    rel_type = RelationType(cardinality)
    target = IdentifierExpression(value=target_str.split("."))
    using_expr = None
    if using_str:
        using_expr = RiverLangParser.parse_expression(using_str.strip())

    return Relation(
        id=rel_id,
        display_name=_id_to_display(rel_id),
        description="",
        target=target,
        relation_type=rel_type,
        using=using_expr,
    )


def _parse_window_line(line: str) -> Window:
    """Parse window: 'win id [partition exprs] [order exprs]'."""
    m = re.match(r"^win\s+(\w+)(.*)?$", line)
    if not m:
        raise CRLParseError(f"Expected window definition: {line}")
    win_id = m.group(1)
    rest = (m.group(2) or "").strip()

    partition_by: list[Expression] = []
    order_by: list[OrderedExpression] = []

    if "partition " in rest:
        pm = re.match(r"^partition\s+(.+?)(?:\s+order\s+(.+))?$", rest)
        if pm:
            for expr_str in _split_group_fields(pm.group(1)):
                partition_by.append(RiverLangParser.parse_expression(
                    _expand_dot_shorthand(expr_str.strip())))
            if pm.group(2):
                order_by = _parse_order_by(pm.group(2))
    elif "order " in rest:
        om = re.match(r"^order\s+(.+)$", rest)
        if om:
            order_by = _parse_order_by(om.group(1))

    return Window(
        id=win_id,
        partition_by=partition_by,
        order_by=order_by,
    )


def _sanitize_expression(expr_str: str) -> str:
    """Fix common LLM expression mistakes before parsing."""
    # Remove inline comments (# ...)
    expr_str = re.sub(r"#[^\n]*", "", expr_str).strip()
    # Remove trailing type annotations like `: float` or `: str`
    expr_str = re.sub(r"\s*:\s*(str|int|float|date|bool|string)\s*$", "", expr_str).strip()
    # Replace && with and, || with or
    expr_str = expr_str.replace("&&", " and ").replace("||", " or ")
    # Remove date() wrapper if LLM added it (not a valid RiverLang function)
    expr_str = re.sub(r'\bdate\("(\d{4}-\d{2}-\d{2})"\)', r'"\1"', expr_str)
    return expr_str


def _expand_dot_shorthand(expr_str: str) -> str:
    """Expand .X shorthand to base.X in expression strings.

    Only expands a leading dot (the whole expression is .X),
    or dots preceded by whitespace/operator that look like shorthand.
    """
    expr_str = expr_str.strip()
    # Simple case: entire expression is .foo
    if expr_str.startswith(".") and not expr_str.startswith(".."):
        return "base" + expr_str
    return expr_str


def _split_semicolons(line: str) -> list[str]:
    """Split a line by semicolons (used for compact multi-field source defs)."""
    parts = [p.strip() for p in line.split(";")]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# CRL split: separate sources (immutable) from entities (editable)
# ---------------------------------------------------------------------------

def split_crl(crl_text: str) -> tuple[str, str, str]:
    """Split CRL into (model_header, sources_section, entities_section).

    Sources are immutable — agent only needs to edit entities.
    Returns:
        model_header: 'model ID {' line
        sources: all src blocks (read-only context)
        entities: all ent/const blocks (editable)
    """
    lines = crl_text.split("\n")
    header = ""
    src_lines: list[str] = []
    ent_lines: list[str] = []
    current = None  # 'src' or 'ent' or None
    depth = 0

    for line in lines:
        stripped = line.strip()

        # Model open/close
        if stripped.startswith("model ") and stripped.endswith("{"):
            header = stripped
            continue
        if stripped == "}" and current is None:
            continue  # closing model brace

        # Detect block start
        if current is None:
            if stripped.startswith("src "):
                current = "src"
                depth = 0
            elif stripped.startswith("ent ") or stripped.startswith("const "):
                current = "ent"
                depth = 0
            elif not stripped:
                continue  # blank line between blocks
            # else: stray line, add to entities
            else:
                ent_lines.append(line)
                continue

        # Accumulate lines in current block
        if current == "src":
            src_lines.append(line)
        else:
            ent_lines.append(line)

        # Track brace depth
        depth += stripped.count("{") - stripped.count("}")
        if depth <= 0:
            current = None

    return header, "\n".join(src_lines), "\n".join(ent_lines)


def merge_crl(header: str, sources: str, entities: str) -> str:
    """Merge sources + entities back into full CRL policy."""
    parts = [header]
    if sources.strip():
        parts.append(sources)
    if entities.strip():
        parts.append(entities)
    parts.append("}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Line numbering and edit application
# ---------------------------------------------------------------------------

def number_lines(code: str) -> str:
    """Add 1-indexed line numbers for LLM display."""
    lines = code.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i + 1:>{width}}|{line}" for i, line in enumerate(lines))


def parse_edits(response: str) -> list[tuple[str, str]]:
    """Parse LLM edit response into (old_string, new_string) pairs.

    Format:
        <<<
        old text
        ===
        new text
        >>>
    """
    edits: list[tuple[str, str]] = []
    blocks = re.split(r"^<<<\s*$", response, flags=re.MULTILINE)
    for block in blocks[1:]:  # skip text before first <<<
        m = re.split(r"^===\s*$", block, flags=re.MULTILINE)
        if len(m) < 2:
            continue
        old_part = m[0].strip("\n")
        rest = m[1]
        # Strip trailing >>>
        rest = re.split(r"^>>>\s*$", rest, flags=re.MULTILINE)[0]
        new_part = rest.strip("\n")
        if old_part or new_part:
            edits.append((old_part, new_part))
    return edits


def apply_edits(code: str, edits: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Apply edit pairs to code. Returns (new_code, list_of_failures)."""
    failures: list[str] = []
    for i, (old, new) in enumerate(edits):
        if not old and not new:
            continue
        if old == new:
            failures.append(f"Edit #{i+1}: no-op (old == new)")
            continue
        if not old:
            # Pure insertion — new text is appended at end
            code = code + "\n" + new
        elif old not in code:
            failures.append(f"Edit #{i+1}: old_string not found: {old[:80]!r}")
        else:
            count = code.count(old)
            if count > 1:
                # Ambiguous — try to apply first match anyway
                code = code.replace(old, new, 1)
            else:
                code = code.replace(old, new, 1)
    return code, failures


def format_compile_errors(crl_code: str, errors: list[str], attempt: int, max_attempts: int) -> str:
    """Format compilation errors with line-numbered code for LLM feedback."""
    numbered = number_lines(crl_code)
    error_list = "\n".join(f"- {e}" for e in errors[:5])
    return (
        f"CRL compilation failed ({len(errors)} errors):\n{error_list}\n\n"
        f"Current CRL:\n```\n{numbered}\n```\n"
        "Fix the errors. All expressions must use base. prefix. Relations need on clause. Respond in CRL."
    )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def is_riverlang(text: str) -> bool:
    """Detect if text is standard RiverLang (not CRL).

    Heuristic: RiverLang uses 'source ' and 'property ' keywords,
    CRL uses 'src ' and inline 'id:type' syntax.
    """
    stripped = text.strip()
    # Check for strong RiverLang indicators
    has_property = "property " in stripped
    has_source = bool(re.search(r"\bsource\s+\w+\(", stripped))
    has_doc = "/**" in stripped
    has_generated = "generated from entity" in stripped
    # Check for CRL indicators
    has_src = bool(re.search(r"\bsrc\s+\w+\s*=", stripped))
    has_ent = bool(re.search(r"\bent\s+\w+\s+from\s+", stripped))

    rl_score = sum([has_property, has_source, has_doc, has_generated])
    crl_score = sum([has_src, has_ent])

    if rl_score > crl_score:
        return True
    if crl_score > 0:
        return False
    # Default: assume RiverLang
    return True
