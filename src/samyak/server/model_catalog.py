"""HTML for Model Intelligence catalog listing. Offline, no JavaScript."""
# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Sequence
from urllib.parse import quote

from samyak.model.catalog import (
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    ModelCatalog,
)
from samyak.model.facts import ContextWindow, Fact, FactStatus, LifecycleState, Modality
from samyak.model.query import ModelQuery, query_models
from samyak.model.records import EvidenceDisposition, ModelRecord, is_currently_documented_active
from samyak.server.layout import escape_html as _e
from samyak.server.layout import format_as_of_date, render_page

PARTIAL_NOTICE_PREFIX = "PARTIAL_MODEL_PAGES"
LIFECYCLE_FILTERS = frozenset({"all", "active", "legacy", "deprecated", "retired", "uncertain"})
UPDATE_COMMAND = "samyak model update openai"

_FRESHNESS_STATUS_LABEL = {
    FreshnessStatus.AS_OF: "As of",
    FreshnessStatus.STALE: "Stale",
    FreshnessStatus.UNKNOWN: "Unknown",
}
_OVERLAY_LABEL = {
    CatalogOverlay.USER_CACHE: "user cache",
    CatalogOverlay.BUNDLED: "bundled",
}
_LIFECYCLE_FILTER_LABEL = {
    "all": "All lifecycle states",
    "active": "Active",
    "legacy": "Legacy",
    "deprecated": "Deprecated",
    "retired": "Retired",
    "uncertain": "Uncertain",
}
_PROVIDER_FILTER_LABEL = {
    "anthropic": "Anthropic",
    "fireworks": "Fireworks",
    "google": "Google",
    "openai": "OpenAI",
    "together": "Together",
}

_EXPLORER_SCRIPT = """
globalThis.samyakEvaluateExplorer=(data,state)=>{let eligible=0;const matching_ids=[],exclusions=[],counts={};const add=r=>{counts[r]=(counts[r]||0)+1};const bool=(x,key,name,expected,reasons)=>{if(!expected)return;if(x[key].status!=='known')reasons.push(name+'_not_verified');else if(String(x[key].value)!==expected)reasons.push(name+'_mismatch')};const modes=(x,key,name,requested,reasons)=>{if(!requested.length)return;if(x[key].status!=='known')reasons.push(name+'_not_verified');else if(!requested.every(v=>x[key].value.includes(v)))reasons.push(name+'_mismatch')};
for(const x of data){if(!x.server_scope)continue;const reasons=[];if(!state.documentation.includes(x.documentation))reasons.push('documentation_not_eligible');else{eligible++;if(state.providers.length&&!state.providers.includes(x.provider))reasons.push('provider_mismatch');if(state.lifecycle){if(x.lifecycle.status!=='known')reasons.push('lifecycle_not_verified');else if(x.lifecycle.value!==state.lifecycle)reasons.push('lifecycle_mismatch')}bool(x,'api','api_access',state.api,reasons);bool(x,'tool','tool_calling',state.tool,reasons);bool(x,'structured','structured_output',state.structured,reasons);modes(x,'input','input_modalities',state.input,reasons);modes(x,'output','output_modalities',state.output,reasons);if(state.context){if(!state.kind||x.context.status!=='known')reasons.push('missing_compatible_context_limit');else if(x.context.value.kind!==state.kind)reasons.push('context_window_kind_mismatch');else if(x.context.value.tokens<state.context)reasons.push('context_window_below_minimum')}}if(reasons.length){exclusions.push({samyak_id:x.samyak_id,reasons});reasons.forEach(add)}else matching_ids.push(x.samyak_id)}return {eligible_count:eligible,match_count:matching_ids.length,matching_ids,eligible,ids:matching_ids,exclusions,exclusion_reason_counts:counts};};
globalThis.samyakExplainExplorer=(data,state,result,modelId)=>{const record=data.find(x=>x.server_scope&&x.samyak_id===modelId);const labels={tool:'Tool calling',api:'API access',structured:'Structured output',input:'Input modalities',output:'Output modalities',lifecycle:'Lifecycle',context:'Context window'};const factLine=(label,fact)=>{if(!fact)return label+': not verified';if(fact.status!=='known')return label+': '+fact.status.replaceAll('_',' ');const value=Array.isArray(fact.value)?fact.value.join(', '):typeof fact.value==='object'?fact.value.tokens+' ('+fact.value.kind+')':String(fact.value);const source=fact.source_ids.length?' Evidence: '+fact.source_ids.join(', ')+'.':'';const verified=fact.verified_at.length?' Verified: '+fact.verified_at.join(', ')+'.':'';return 'Verified '+label.toLowerCase()+': '+value+'.'+source+verified};if(!record)return {summary:'Choose a model in this page to inspect its recorded evidence.',lines:[],href:null};const matching=result.matching_ids.includes(record.samyak_id);const lines=[];if(!state.documentation.includes(record.documentation))lines.push('Excluded by documentation filter: '+record.documentation.replaceAll('_',' ')+'.');if(state.providers.length&&!state.providers.includes(record.provider))lines.push('Excluded by provider filter: '+record.provider+'.');for(const key of ['lifecycle','tool','api','structured','input','output','context']){const selected=key==='lifecycle'?state.lifecycle:key==='tool'?state.tool:key==='api'?state.api:key==='structured'?state.structured:key==='input'?state.input.length:key==='output'?state.output.length:state.context;if(selected)lines.push(factLine(labels[key],record[key]));}if(!lines.length)lines.push('No capability predicate is selected; this view uses the documentation filter.');return {summary:matching?'This model matches the selected verified-evidence filters.':'This model is excluded by one or more selected filters. Missing evidence is not a verified negative.',lines,href:record.detail_href};};
globalThis.samyakCompareExplorer=(data,ids,options={})=>{if(ids.length<2||ids.length>3)throw new Error('compare exactly two or three model IDs');if(new Set(ids).size!==ids.length)throw new Error('comparison model IDs must be unique');const allowed=new Set(['current_documented',...(options.documentation||[])]);if(options.include_legacy)allowed.add('legacy_uncertified');const selected=ids.map(id=>{const x=data.find(item=>item.server_scope&&item.samyak_id===id);if(!x)throw new Error('unknown comparison model ID: '+id);if(!allowed.has(x.documentation))throw new Error(id+' has documentation disposition '+x.documentation+'; opt in explicitly');return x});const keys={api_access:'api',tool_calling:'tool',structured_output:'structured',input_modalities:'input',output_modalities:'output',context_window:'context'};const fact=(name,x)=>{const item=x[keys[name]||name];return {name,status:item.status,value:item.value,source_ids:item.source_ids,verified_at:item.verified_at,claims:item.claims||[]}};const names=['display_name','lifecycle','availability_scope','api_access','tool_calling','structured_output','input_modalities','output_modalities','context_window'];const models=selected.map(x=>({samyak_id:x.samyak_id,provider_id:x.provider,documentation_disposition:x.documentation,detail_href:x.detail_href,facts:Object.fromEntries(names.map(name=>[name,fact(name,x)]))}));const known=name=>models.every(m=>m.facts[name].status==='known');const dimension=name=>({name,comparable:known(name),reason:known(name)?null:'not all selected records have verified evidence'});const context=(()=>{const facts=models.map(m=>m.facts.context_window);if(!facts.every(f=>f.status==='known'&&f.value&&f.value.kind))return {name:'context_window',comparable:false,reason:'not all selected records have context evidence'};const kinds=new Set(facts.map(f=>f.value.kind));return kinds.size===1&&!kinds.has('unknown')?{name:'context_window',comparable:true,reason:null}:{name:'context_window',comparable:false,reason:'context kinds are not directly comparable'}})();const availability=(()=>{const facts=models.map(m=>m.facts.availability_scope);if(!facts.every(f=>f.status==='known'))return {name:'availability_scope',comparable:false,reason:'availability scope is not verified for all'};return new Set(facts.map(f=>f.value)).size===1?{name:'availability_scope',comparable:true,reason:null}:{name:'availability_scope',comparable:false,reason:'availability scopes differ'}})();const dispositions=[...new Set(models.map(m=>m.documentation_disposition))].sort();return {models,dimensions:['api_access','tool_calling','structured_output','input_modalities','output_modalities','lifecycle'].map(dimension).concat([context,availability]),warnings:dispositions.length===1&&dispositions[0]==='current_documented'?[]:['comparison includes multiple documentation certification levels: '+dispositions.join(', ')]};};
if(typeof document!=='undefined'){const bind=()=>{const d=JSON.parse(document.getElementById('model-query-index').textContent),q=k=>document.querySelector('[data-q='+k+']'),vals=k=>[...document.querySelectorAll('[data-q='+k+']:checked,[data-q='+k+'] option:checked')].map(x=>x.value),rows=new Map;document.querySelectorAll('[data-model-id]').forEach(cell=>{const id=cell.dataset.modelId;if(rows.has(id))throw new Error('duplicate rendered model id: '+id);rows.set(id,cell.parentElement)});const state=()=>({documentation:vals('documentation'),providers:vals('provider'),lifecycle:q('lifecycle').value,tool:q('tool').value,api:q('api').value,structured:q('structured').value,input:vals('input'),output:vals('output'),context:Number(q('context').value),kind:q('kind').value});function explain(r,s){const e=globalThis.samyakExplainExplorer(d,s,r,q('inspect').value),panel=q('explanation');panel.replaceChildren();const summary=document.createElement('p');summary.textContent=e.summary;panel.append(summary);const list=document.createElement('ul');for(const line of e.lines){const item=document.createElement('li');item.textContent=line;list.append(item)}panel.append(list);if(e.href){const link=document.createElement('a');link.href=e.href;link.textContent='Open model evidence details';panel.append(link)}}function run(){const s=state(),r=globalThis.samyakEvaluateExplorer(d,s),matches=new Set(r.matching_ids);for(const id of r.matching_ids)if(!rows.has(id))throw new Error('missing rendered model id: '+id);rows.forEach((row,id)=>{row.hidden=!matches.has(id)});const categories=Object.entries(r.exclusion_reason_counts).map(([reason,count])=>count+' '+reason.replaceAll('_',' ')).join('; ');q('count').textContent=r.match_count+' matches from '+r.eligible_count+' documentation-eligible records. '+(r.match_count?'':'No verified matching records. '+categories+'. Missing evidence or a zero result does not prove no model is capable. Exclusion categories can overlap.');explain(r,s)}document.querySelectorAll('.capability-controls [data-q]').forEach(x=>x.addEventListener('input',run));q('inspect').addEventListener('input',run);q('clear').addEventListener('click',()=>{document.querySelectorAll('.capability-controls select,.capability-controls input').forEach(x=>{if(x.type==='checkbox')x.checked=false;else x.value=''});document.querySelector('[data-q=documentation]').checked=true;run()});run()};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',bind,{once:true});else bind();}
""".strip()

_COMPARISON_UI_SCRIPT = """
if (typeof document !== 'undefined') {
  const bindCompare = () => {
    const q = key => document.querySelector('[data-compare=' + key + ']');
    const data = JSON.parse(document.getElementById('model-query-index').textContent);
    const chosen = [];
    const announce = message => { q('status').textContent = message; };
    const options = () => ({
      include_legacy: q('legacy').checked,
      documentation: [...document.querySelectorAll('[data-compare-doc]:checked')].map(input => input.value),
    });
    const label = name => name.replaceAll('_', ' ');
    const valueText = value => typeof value === 'string' ? value : JSON.stringify(value);

    function appendRow(table, heading, models, renderCell) {
      const row = document.createElement('tr');
      const labelCell = document.createElement('th');
      labelCell.scope = 'row';
      labelCell.textContent = heading;
      row.append(labelCell);
      for (const model of models) row.append(renderCell(model));
      table.append(row);
    }

    function evidenceDetails(fact) {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'Evidence';
      details.append(summary);
      const evidence = document.createElement('p');
      evidence.textContent = 'Sources: ' + (fact.source_ids.join(', ') || 'no source reference recorded')
        + '; verified: ' + (fact.verified_at.join(', ') || 'no verification timestamp');
      details.append(evidence);
      if (fact.claims.length) {
        const claims = document.createElement('ul');
        for (const claim of fact.claims) {
          const item = document.createElement('li');
          item.textContent = 'Claim value: ' + valueText(claim.value)
            + '; source ID: ' + (claim.source_id || 'no source reference recorded')
            + '; verified: ' + (claim.verified_at || 'no verification timestamp');
          claims.append(item);
        }
        details.append(claims);
      }
      return details;
    }

    function factCell(fact) {
      const cell = document.createElement('td');
      const state = document.createElement('span');
      state.textContent = fact.status === 'known'
        ? 'Verified ' + valueText(fact.value)
        : fact.status.replaceAll('_', ' ');
      cell.append(state, evidenceDetails(fact));
      return cell;
    }

    function render() {
      const root = q('result');
      root.replaceChildren();
      q('count').textContent = chosen.length + ' selected';
      if (chosen.length < 2) {
        announce('Select two or three models to compare.');
        return;
      }
      let result;
      try {
        result = globalThis.samyakCompareExplorer(data, chosen, options());
      } catch (error) {
        announce(error.message);
        return;
      }
      announce('Comparison ready for ' + chosen.length + ' models.');
      const table = document.createElement('table');
      const header = document.createElement('tr');
      header.append(document.createElement('th'));
      for (const model of result.models) {
        const cell = document.createElement('th');
        cell.scope = 'col';
        cell.textContent = model.samyak_id;
        header.append(cell);
      }
      table.append(header);
      appendRow(table, 'Provider', result.models, model => {
        const cell = document.createElement('td');
        cell.textContent = model.provider_id;
        return cell;
      });
      appendRow(table, 'Documentation disposition', result.models, model => {
        const cell = document.createElement('td');
        cell.textContent = model.documentation_disposition;
        return cell;
      });
      appendRow(table, 'Stable model ID', result.models, model => {
        const cell = document.createElement('td');
        cell.textContent = model.samyak_id;
        return cell;
      });
      appendRow(table, 'Display name', result.models, model => factCell(model.facts.display_name));
      appendRow(table, 'Model details', result.models, model => {
        const cell = document.createElement('td');
        const link = document.createElement('a');
        link.href = model.detail_href;
        link.textContent = 'Open model evidence details';
        cell.append(link);
        return cell;
      });
      for (const name of ['lifecycle', 'availability_scope', 'api_access', 'tool_calling', 'structured_output', 'input_modalities', 'output_modalities', 'context_window']) {
        appendRow(table, label(name), result.models, model => factCell(model.facts[name]));
      }
      root.append(table);
      for (const item of result.dimensions) {
        if (!item.comparable) {
          const paragraph = document.createElement('p');
          paragraph.textContent = item.name + ': not directly comparable — ' + item.reason;
          root.append(paragraph);
        }
      }
      for (const warning of result.warnings) {
        const paragraph = document.createElement('p');
        paragraph.textContent = warning;
        root.append(paragraph);
      }
    }

    document.querySelectorAll('[data-compare-select]').forEach(box => box.addEventListener('change', () => {
      const id = box.dataset.compareSelect;
      if (box.checked) {
        if (chosen.length === 3) {
          box.checked = false;
          announce('Only three models can be compared.');
          return;
        }
        chosen.push(id);
      } else {
        const index = chosen.indexOf(id);
        if (index >= 0) chosen.splice(index, 1);
      }
      render();
    }));
    q('clear').addEventListener('click', () => {
      chosen.splice(0);
      document.querySelectorAll('[data-compare-select]').forEach(box => { box.checked = false; });
      render();
    });
    document.querySelectorAll('[data-compare-doc], [data-compare=legacy]').forEach(input => input.addEventListener('change', render));
    render();
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bindCompare, { once: true });
  else bindCompare();
}
""".strip()


def model_href(samyak_id: str) -> str:
    return "/models/" + quote(samyak_id, safe=":@._-")


def format_token_count(tokens: int) -> str:
    if tokens >= 1000 and tokens % 1000 == 0:
        thousands = tokens // 1000
        if thousands >= 1000 and thousands % 1000 == 0:
            return f"{thousands // 1000}M"
        return f"{thousands}K"
    return str(tokens)


def lifecycle_matches(record: ModelRecord, lifecycle: str) -> bool:
    if lifecycle == "all":
        return True
    fact = record.lifecycle
    if lifecycle == "uncertain":
        return fact.status is not FactStatus.KNOWN
    if lifecycle == "active":
        return is_currently_documented_active(record)
    if fact.status is not FactStatus.KNOWN:
        return False
    value = fact.value
    return isinstance(value, LifecycleState) and value.value == lifecycle


def search_matches(record: ModelRecord, query: str) -> bool:
    needle = query.strip().lower()
    if not needle:
        return True
    haystacks = [record.samyak_id, record.provider_model_id, record.provider_id]
    if record.display_name.status is FactStatus.KNOWN and isinstance(
        record.display_name.value, str
    ):
        haystacks.append(record.display_name.value)
    if record.aliases.status is FactStatus.KNOWN and record.aliases.value:
        haystacks.extend(record.aliases.value)
    return any(needle in item.lower() for item in haystacks)


def provider_matches(record: ModelRecord, provider: str) -> bool:
    if provider == "all":
        return True
    return record.provider_id == provider


def filter_models(
    models: Sequence[ModelRecord],
    *,
    q: str = "",
    lifecycle: str = "all",
    provider: str = "all",
    provider_ids: Sequence[str] | None = None,
) -> tuple[ModelRecord, ...]:
    chosen_lifecycle = lifecycle if lifecycle in LIFECYCLE_FILTERS else "all"
    known_providers = (
        provider_ids
        if provider_ids is not None
        else tuple(dict.fromkeys(record.provider_id for record in models))
    )
    chosen_provider = _normalize_provider_filter(provider, known_providers)
    return tuple(
        record
        for record in models
        if lifecycle_matches(record, chosen_lifecycle)
        and provider_matches(record, chosen_provider)
        and search_matches(record, q)
    )


def catalog_is_partial(notices: Sequence[CatalogNotice]) -> bool:
    return any(notice.code.startswith(PARTIAL_NOTICE_PREFIX) for notice in notices)


def lifecycle_counts(models: Sequence[ModelRecord]) -> dict[str, int]:
    counts = {state.value: 0 for state in LifecycleState}
    counts["uncertain"] = 0
    for record in models:
        fact = record.lifecycle
        if fact.status is FactStatus.KNOWN and isinstance(fact.value, LifecycleState):
            if fact.value is LifecycleState.ACTIVE and not is_currently_documented_active(record):
                counts["uncertain"] += 1
                continue
            counts[fact.value.value] += 1
        else:
            counts["uncertain"] += 1
    return counts


def render_freshness_block(freshness: CatalogFreshness, *, compact: bool = False) -> str:
    status = _FRESHNESS_STATUS_LABEL.get(freshness.status, freshness.status.value)
    as_of = format_as_of_date(freshness.generated_at)
    overlay = _OVERLAY_LABEL.get(freshness.overlay, freshness.overlay.value)
    if compact:
        return f'<p class="muted">{_e(status)} {_e(as_of)}</p>'
    rows = [
        f"<div><dt>Status</dt><dd>{_e(status)}</dd></div>",
        f"<div><dt>As of</dt><dd>{_e(as_of)}</dd></div>",
        f"<div><dt>Overlay</dt><dd>{_e(overlay)}</dd></div>",
    ]
    if freshness.oldest_verified_at:
        rows.append(
            "<div><dt>Oldest evidence</dt>"
            f"<dd>{_e(format_as_of_date(freshness.oldest_verified_at))}</dd></div>"
        )
    if freshness.stale_after:
        rows.append(
            "<div><dt>Labeled stale after</dt>"
            f"<dd>{_e(format_as_of_date(freshness.stale_after))}</dd></div>"
        )
    return f'<dl class="meta">{"".join(rows)}</dl>'


def render_notices(notices: Sequence[CatalogNotice]) -> str:
    if not notices:
        return ""
    items = "".join(
        f"<li><code>{_e(notice.code)}</code> {_e(notice.message)}</li>" for notice in notices
    )
    notice_list = f'<ul class="notices">{items}</ul>'
    refresh = _refresh_html()
    if catalog_is_partial(notices):
        explanation = (
            '<p class="banner warn">This catalog is incomplete. Some model pages '
            "could not be retrieved, and facts from those pages are omitted.</p>"
        )
        return (
            '<details class="catalog-health">'
            "<summary>Catalog is partial "
            '<span class="muted">Some provider documentation could not be retrieved.</span>'
            "</summary>"
            f"{explanation}{notice_list}{refresh}"
            "</details>"
        )
    return (
        '<details class="catalog-health">'
        "<summary>Catalog notices</summary>"
        f"{notice_list}{refresh}"
        "</details>"
    )


def render_catalog_missing_page() -> str:
    return render_page(
        title="Model Catalog — Samyak",
        current="/models",
        extra_css=_CSS,
        body=_missing_body(),
    )


def render_catalog_unavailable_page(*, unsupported_schema: bool) -> str:
    if unsupported_schema:
        title = "Unsupported catalog schema — Samyak"
        heading = "Unsupported catalog schema"
        message = (
            "This catalog cannot be read by this Samyak version. "
            f"Update Samyak, or replace the catalog with <code>{_e(UPDATE_COMMAND)}</code>."
        )
    else:
        title = "Model catalog unavailable — Samyak"
        heading = "Model catalog unavailable"
        message = (
            "The local model catalog could not be read. It may be corrupt. "
            f"Replace it with <code>{_e(UPDATE_COMMAND)}</code>."
        )
    return render_page(
        title=title,
        current="/models",
        extra_css=_CSS,
        body="".join(
            [
                "<header>",
                '<p class="eyebrow">Samyak</p>',
                f"<h1>{_e(heading)}</h1>",
                f"<p>{message}</p>",
                '<p><a href="/">Workspace</a></p>',
                "</header>",
            ]
        ),
    )


def render_model_catalog(
    catalog: ModelCatalog,
    *,
    q: str = "",
    lifecycle: str = "all",
    provider: str = "all",
) -> str:
    chosen_lifecycle = lifecycle if lifecycle in LIFECYCLE_FILTERS else "all"
    provider_ids = tuple(item.id for item in catalog.providers)
    chosen_provider = _normalize_provider_filter(provider, provider_ids)
    filtered = filter_models(
        catalog.models,
        q=q,
        lifecycle=chosen_lifecycle,
        provider=chosen_provider,
        provider_ids=provider_ids,
    )
    body = "".join(
        [
            "<header>",
            '<p class="eyebrow">Samyak</p>',
            "<h1>Model Catalog</h1>",
            '<p class="lede">Serving-source model information verified from captured '
            "documentation.</p>",
            "</header>",
            "<section>",
            "<h2>Catalog</h2>",
            render_freshness_block(catalog.freshness),
            _provider_line(catalog),
            render_notices(catalog.notices) or _refresh_html(),
            "</section>",
            "<section>",
            "<h2>Models</h2>",
            _capability_explorer(catalog, filtered),
            _filter_form(
                q=q,
                lifecycle=chosen_lifecycle,
                provider=chosen_provider,
                provider_ids=provider_ids,
            ),
            _catalog_table(catalog, filtered),
            "</section>",
        ]
    )
    return render_page(
        title="Model Catalog — Samyak",
        current="/models",
        extra_css=_CSS,
        body=body,
    )


def _capability_explorer(catalog: ModelCatalog, server_scope: Sequence[ModelRecord]) -> str:
    """Embed a compact typed-fact index; no provider parsing runs in browser."""
    _ensure_unique_model_ids(server_scope)
    strict = query_models(server_scope, ModelQuery())
    index = [
        _query_index_record(record)
        for record in sorted(server_scope, key=lambda item: item.samyak_id)
    ]
    payload = json.dumps(index, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    inspect_options = "".join(
        f'<option value="{_e(record.samyak_id)}">{_e(record.samyak_id)}</option>'
        for record in sorted(server_scope, key=lambda item: item.samyak_id)
    )
    return (
        '<details class="capability-explorer"><summary>Capability explorer</summary>'
        '<p class="muted">Filters use verified catalog facts. Default: currently documented only; '
        "unknown, unverified, and conflicting facts never match.</p>"
        '<div class="capability-controls">'
        '<label><input type="checkbox" data-q="documentation" value="current_documented" checked> Current documented</label>'
        '<label><input type="checkbox" data-q="documentation" value="legacy_uncertified"> Add legacy</label>'
        '<label><input type="checkbox" data-q="documentation" value="historical_reference_only"> Add historical</label>'
        '<label><input type="checkbox" data-q="documentation" value="not_observed_in_current_capture"> Add not observed</label>'
        '<label>Provider <select data-q="provider" multiple><option value="anthropic">Anthropic</option><option value="fireworks">Fireworks</option><option value="google">Google</option><option value="openai">OpenAI</option><option value="together">Together</option></select></label>'
        '<label>Lifecycle <select data-q="lifecycle"><option value="">Any</option><option value="active">Active</option><option value="legacy">Legacy</option><option value="deprecated">Deprecated</option><option value="retired">Retired</option></select></label>'
        '<label>Tool calling <select data-q="tool"><option value="">Any</option>'
        '<option value="true">True</option><option value="false">False</option></select></label>'
        '<label>API access <select data-q="api"><option value="">Any</option>'
        '<option value="true">True</option><option value="false">False</option></select></label>'
        '<label>Structured output <select data-q="structured"><option value="">Any</option><option value="true">True</option><option value="false">False</option></select></label>'
        '<label>Input modalities <select data-q="input" multiple><option value="text">Text</option><option value="image">Image</option><option value="audio">Audio</option><option value="video">Video</option></select></label>'
        '<label>Output modalities <select data-q="output" multiple><option value="text">Text</option><option value="image">Image</option><option value="audio">Audio</option><option value="video">Video</option></select></label>'
        '<label>Minimum context <input data-q="context" type="number" min="1"></label>'
        '<label>Context kind <select data-q="kind"><option value="">Select kind</option>'
        '<option value="input">Input</option><option value="output">Output</option>'
        '<option value="combined">Combined</option></select></label>'
        '<button type="button" data-q="clear">Clear filters</button></div>'
        f'<p class="muted" data-q="count">Strict evidence policy: '
        f"{strict.summary.matching_records} matches from "
        f"{strict.summary.documentation_eligible_records} eligible records.</p>"
        '<section class="capability-explanation" aria-labelledby="capability-explanation-title">'
        '<h3 id="capability-explanation-title">Why this result</h3>'
        '<label>Inspect a model in this page <select data-q="inspect">'
        '<option value="">Choose a model</option>'
        f"{inspect_options}</select></label>"
        '<div data-q="explanation" aria-live="polite"></div></section>'
        f'<script type="application/json" id="model-query-index">{payload}</script>'
        f"<script>{_EXPLORER_SCRIPT}</script>"
        f'<section class="comparison-workspace"><h3>Compare models</h3><p data-compare="count">0 selected</p><label><input type="checkbox" data-compare="legacy"> Include legacy</label><label><input type="checkbox" data-compare-doc value="historical_reference_only"> Include historical</label><button type="button" data-compare="clear">Clear selection</button><p data-compare="status" aria-live="polite"></p><div data-compare="result"></div></section>'
        f"<script>{_COMPARISON_UI_SCRIPT}</script></details>"
    )


def _ensure_unique_model_ids(records: Sequence[ModelRecord]) -> None:
    ids = [record.samyak_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("capability explorer requires unique samyak_id values")


def _query_index_record(record: ModelRecord) -> dict[str, object]:
    def compact(fact: Fact) -> dict[str, object]:
        value = fact.value
        if isinstance(value, ContextWindow):
            value = value.to_dict()
        elif isinstance(value, tuple):
            value = [item.value if isinstance(item, Modality) else item for item in value]
        provenances = []
        if fact.provenance is not None:
            provenances.append(fact.provenance)
        provenances.extend(claim.provenance for claim in fact.claims)
        return {
            "status": fact.status.value,
            "value": value,
            "source_ids": [item.source_id for item in provenances],
            "verified_at": [item.verified_at for item in provenances if item.verified_at],
            "claims": [
                {
                    "value": claim.value.to_dict()
                    if isinstance(claim.value, ContextWindow)
                    else claim.value,
                    "source_id": claim.provenance.source_id,
                    "verified_at": claim.provenance.verified_at,
                }
                for claim in fact.claims
            ],
        }

    return {
        "samyak_id": record.samyak_id,
        "detail_href": model_href(record.samyak_id),
        "server_scope": True,
        "documentation": "legacy_uncertified"
        if record.evidence_disposition is None
        else record.evidence_disposition.value,
        "provider": record.provider_id,
        "display_name": compact(record.display_name),
        "availability_scope": compact(record.availability_scope),
        "lifecycle": compact(record.lifecycle),
        "tool": compact(record.tool_calling),
        "api": compact(record.api_access),
        "structured": compact(record.structured_output),
        "input": compact(record.input_modalities),
        "output": compact(record.output_modalities),
        "context": compact(record.context_window),
    }


def _missing_body() -> str:
    return "".join(
        [
            "<header>",
            '<p class="eyebrow">Samyak</p>',
            "<h1>Model Catalog</h1>",
            '<p class="lede">Serving-source model information verified from captured '
            "documentation.</p>",
            "</header>",
            '<p class="empty">No local model catalog yet. Model Intelligence becomes '
            "available after "
            f"<code>{_e(UPDATE_COMMAND)}</code>. Then refresh this page. "
            "Browsing stays offline.</p>",
        ]
    )


def _provider_line(catalog: ModelCatalog) -> str:
    if not catalog.providers:
        return ""
    parts = [f"{_e(item.id)}: {_e(item.model_count)}" for item in catalog.providers]
    return f'<p class="muted">Serving sources: {", ".join(parts)}</p>'


def _refresh_html() -> str:
    return (
        f'<p class="muted">Refresh with <code>{_e(UPDATE_COMMAND)}</code>, '
        "<code>samyak model update anthropic</code>, "
        "<code>samyak model update google</code>, "
        "<code>samyak model update fireworks</code>, or "
        "<code>samyak model update together</code>.</p>"
    )


def _normalize_provider_filter(provider: str, provider_ids: Sequence[str]) -> str:
    chosen = provider.strip().lower()
    if chosen == "all" or chosen in provider_ids:
        return chosen or "all"
    return "all"


def _provider_filter_label(provider_id: str) -> str:
    return _PROVIDER_FILTER_LABEL.get(provider_id, provider_id)


def _filter_form(
    *,
    q: str,
    lifecycle: str,
    provider: str,
    provider_ids: Sequence[str],
) -> str:
    lifecycle_options = []
    for value in ("all", "active", "legacy", "deprecated", "retired", "uncertain"):
        selected = " selected" if value == lifecycle else ""
        lifecycle_options.append(
            f'<option value="{_e(value)}"{selected}>{_e(_LIFECYCLE_FILTER_LABEL[value])}</option>'
        )
    provider_options = [
        f'<option value="all"{" selected" if provider == "all" else ""}>All providers</option>'
    ]
    for provider_id in provider_ids:
        selected = " selected" if provider_id == provider else ""
        provider_options.append(
            f'<option value="{_e(provider_id)}"{selected}>'
            f"{_e(_provider_filter_label(provider_id))}</option>"
        )
    return (
        '<form class="filters" method="get" action="/models">'
        '<label>Search <input type="search" name="q" '
        f'value="{_e(q)}" aria-label="Search models"></label>'
        "<label>Provider "
        f'<select name="provider" aria-label="Provider">{"".join(provider_options)}</select>'
        "</label>"
        "<label>Lifecycle "
        f'<select name="lifecycle" aria-label="Lifecycle">{"".join(lifecycle_options)}</select>'
        "</label>"
        '<button type="submit">Apply</button>'
        "</form>"
    )


def _catalog_table(catalog: ModelCatalog, models: Sequence[ModelRecord]) -> str:
    if not catalog.models:
        return '<p class="empty">This catalog contains no models.</p>'
    if not models:
        return '<p class="empty">No models match these filters.</p>'
    positions = {record.samyak_id: index for index, record in enumerate(catalog.models)}
    rows = "".join(_model_row(record, positions[record.samyak_id]) for record in models)
    hint = '<p class="scroll-hint muted">On a narrow screen, scroll the table sideways.</p>'
    return (
        f"{hint}"
        '<div class="table-wrap">'
        '<table class="models">'
        "<thead><tr><th>Compare</th><th>Model</th><th>Documentation</th><th>Lifecycle</th><th>Context</th>"
        "<th>Max input</th><th>Max output</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
    )


def _model_row(record: ModelRecord, index: int) -> str:
    name = _model_label(record)
    href = model_href(record.samyak_id)
    return (
        "<tr>"
        f'<td><label><input type="checkbox" data-compare-select="{_e(record.samyak_id)}"> Compare {_e(record.samyak_id)}</label></td>'
        f'<td data-model-index="{index}" data-model-id="{_e(record.samyak_id)}">'
        f'<a href="{_e(href)}">{_e(name)}</a>'
        f'<div class="muted id">{_e(record.samyak_id)}</div>'
        "</td>"
        f"<td>{evidence_disposition_html(record)}</td>"
        f"<td>{lifecycle_html(record)}</td>"
        f"<td>{context_html(record)}</td>"
        f"<td>{token_limit_html(record.max_input_tokens)}</td>"
        f"<td>{token_limit_html(record.max_output_tokens)}</td>"
        "</tr>"
    )


def evidence_disposition_label(record: ModelRecord) -> str:
    """Describe documentation coverage without making an availability claim."""
    labels = {
        EvidenceDisposition.CURRENT_DOCUMENTED: "Currently documented",
        EvidenceDisposition.HISTORICAL_REFERENCE_ONLY: "Historical reference only",
        EvidenceDisposition.NOT_OBSERVED_IN_CURRENT_CAPTURE: (
            "Not observed in latest complete capture"
        ),
        None: "Documentation status not certified",
    }
    return labels[record.evidence_disposition]


def evidence_disposition_html(record: ModelRecord) -> str:
    disposition = record.evidence_disposition
    css_class = (
        "documentation-uncertified" if disposition is None else f"documentation-{disposition.value}"
    )
    return (
        f'<span class="documentation {css_class}">{_e(evidence_disposition_label(record))}</span>'
    )


def _model_label(record: ModelRecord) -> str:
    if record.display_name.status is FactStatus.KNOWN and isinstance(
        record.display_name.value, str
    ):
        return record.display_name.value
    return record.provider_model_id


def lifecycle_html(record: ModelRecord) -> str:
    fact = record.lifecycle
    if fact.status is FactStatus.KNOWN and isinstance(fact.value, LifecycleState):
        label = fact.value.value.capitalize()
        return f'<span class="life life-{_e(fact.value.value)}">{_e(label)}</span>'
    return _status_html(fact.status)


def context_html(record: ModelRecord) -> str:
    fact = record.context_window
    if fact.status is FactStatus.KNOWN and isinstance(fact.value, ContextWindow):
        return _e(format_token_count(fact.value.tokens))
    return _status_html(fact.status)


def token_limit_html(fact: Fact[int]) -> str:
    if (
        fact.status is FactStatus.KNOWN
        and isinstance(fact.value, int)
        and not isinstance(fact.value, bool)
    ):
        return _e(format_token_count(fact.value))
    return _status_html(fact.status)


def _status_html(status: FactStatus) -> str:
    label = _e(_status_label(status))
    if status is FactStatus.UNKNOWN:
        return f'<span class="fact-unknown">{label}</span>'
    if status is FactStatus.NOT_VERIFIED:
        return f'<span class="fact-not-verified">{label}</span>'
    if status is FactStatus.CONFLICT:
        return f'<span class="fact-conflict">{label}</span>'
    return f'<span class="muted">{label}</span>'


def _status_label(status: FactStatus) -> str:
    labels = {
        FactStatus.UNKNOWN: "Unknown",
        FactStatus.NOT_VERIFIED: "Not verified",
        FactStatus.NOT_APPLICABLE: "Not applicable",
        FactStatus.CONFLICT: "Conflict",
        FactStatus.KNOWN: "Known",
    }
    return labels.get(status, status.value)


_CSS = (
    "dl.meta{display:grid;gap:.35rem 1rem;margin:.5rem 0 1rem}"
    "dl.meta>div{display:grid;grid-template-columns:minmax(8rem,28%) 1fr;gap:.35rem 1rem}"
    "dl.meta dt{font-weight:600;color:#5c574f}"
    "dl.meta dd{margin:0}"
    "ul.notices{margin:.75rem 0 0;padding-left:1.2rem}"
    ".catalog-health{margin:.5rem 0 .75rem;padding:.1rem 0 .1rem .7rem;"
    "border-left:3px solid #d9b08c}"
    ".catalog-health summary{cursor:pointer;font-weight:650;color:#1d3f6e;"
    "display:flex;flex-wrap:wrap;gap:.15rem .65rem;align-items:baseline}"
    ".catalog-health summary .muted{font-weight:400}"
    ".catalog-health .banner,.catalog-health ul.notices,.catalog-health p.muted"
    "{margin:.5rem 0 0}"
    ".fact-unknown{color:#8a857c;font-style:italic}"
    ".fact-not-verified{color:#5c574f;font-weight:600;"
    "border-bottom:1px dotted #8a857c}"
    ".fact-conflict{color:#6d4a00;background:#fbf4ea;padding:.05rem .3rem}"
    "form.filters{display:flex;flex-wrap:wrap;gap:.75rem 1rem;align-items:end;margin:0 0 1rem}"
    "form.filters label{display:grid;gap:.25rem;font-size:.92rem;color:#5c574f}"
    "form.filters input,form.filters select,form.filters button{font:inherit}"
    "form.filters input,form.filters select{padding:.3rem .45rem;min-width:10rem}"
    "form.filters button{padding:.4rem .8rem;background:#1d3f6e;color:#fff;border:0;cursor:pointer}"
    "table.models{min-width:48rem;width:100%;border-collapse:collapse;font-size:.92rem}"
    "table.models th,table.models td{text-align:left;padding:.55rem .7rem;"
    "border-bottom:1px solid #ece7de;vertical-align:top}"
    "table.models th{font-weight:600}"
    "table.models .id{margin:.15rem 0 0;word-break:break-all}"
    ".life{display:inline-block;padding:.1rem .45rem;font-size:.75rem;"
    "font-weight:700;letter-spacing:.04em}"
    ".life-active{background:#d5e4f5;color:#1d3f6e}"
    ".life-legacy{background:#e8e1f4;color:#3d2b66}"
    ".life-deprecated{background:#f7e1b5;color:#6d4a00}"
    ".life-retired{background:#e3e3e3;color:#333}"
    ".documentation{font-size:.82rem;line-height:1.3}"
    ".documentation-current_documented{color:#1d3f6e}"
    ".documentation-historical_reference_only{color:#6d4a00;font-weight:600}"
    ".documentation-not_observed_in_current_capture{color:#5c574f;font-weight:600}"
    ".documentation-uncertified{color:#5c574f;font-style:italic}"
    ".comparison-workspace{width:100%;max-width:100%;overflow-x:auto;"
    "-webkit-overflow-scrolling:touch}"
    ".comparison-workspace table{min-width:max-content;border-collapse:collapse}"
    ".scroll-hint{display:none;margin:.35rem 0 .6rem}"
    "@media (max-width:40rem){.scroll-hint{display:block}}"
)
