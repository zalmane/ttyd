"""CRL grammar specification and edit format strings for agent prompts."""

CRL_GRAMMAR_SPEC = """The policy uses CRL (Compact RiverLang). Same semantics as RiverLang but shorter syntax:
- `src` not `source`, `ent` not `entity`, `rel` not `relation`, `str` not `string`
- No display names or doc comments. Types: str int float date bool
- Annotations: @key @dim @met
- Properties: id:type = expression (expressions use base.X, .X shorthand for base.X)
- Relations: rel id -> one|many MODEL.ENTITY on base.X == target.Y (on clause REQUIRED, use 1==1 for cross-join)
- All expressions MUST use base. prefix (or .X shorthand)

The sources (src blocks) below are READ-ONLY context. Do NOT include them in your output.
Only output the entity definitions (ent blocks). Respond in CRL format."""
