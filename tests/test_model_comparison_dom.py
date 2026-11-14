"""DOM interaction coverage for the static model-comparison workspace."""
# ruff: noqa: E501

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_model_viewer import _catalog, _conflict, _known, _model, _sample_catalog

from samyak.model.facts import ContextWindow, ContextWindowKind, LifecycleState, Modality
from samyak.model.records import EvidenceDisposition
from samyak.model.store import FileCatalogStore
from samyak.server.model_catalog import model_href, render_model_catalog


def _run_dom(html: str) -> dict[str, object]:
    program = r"""
const {JSDOM}=require('jsdom');
let html='';process.stdin.on('data',x=>html+=x);process.stdin.on('end',()=>{
 const dom=new JSDOM(html,{runScripts:'dangerously'});const d=dom.window.document;d.dispatchEvent(new dom.window.Event('DOMContentLoaded'));
 const boxes=[...d.querySelectorAll('[data-compare-select]')];
 const click=i=>{boxes[i].checked=true;boxes[i].dispatchEvent(new dom.window.Event('change',{bubbles:true}))};
 const state=label=>{const result=d.querySelector('[data-compare=result]');return {label,count:d.querySelector('[data-compare=count]').textContent,status:d.querySelector('[data-compare=status]').textContent,tables:result.querySelectorAll('table').length,headers:[...result.querySelectorAll('th')].map(x=>x.textContent),text:result.textContent,links:[...result.querySelectorAll('a')].map(x=>({href:x.getAttribute('href'),text:x.textContent})),scripts:result.querySelectorAll('script').length}};
 const out=[state('empty')];click(0);out.push(state('one'));click(1);out.push(state('two'));click(2);out.push(state('three'));if(boxes.length>3){click(3);out.push(state('four'))}d.querySelector('[data-compare=clear]').click();out.push(state('clear'));
 process.stdout.write(JSON.stringify({out,labels:boxes.map(x=>x.parentElement.textContent),scripts:dom.window.samyakCompareExplorer!==undefined}));
});"""
    env = {**os.environ, "NODE_PATH": "/private/tmp/samyak-dom-tests/node_modules"}
    result = subprocess.run(
        ["node", "-e", program], input=html, text=True, capture_output=True, check=True, env=env
    )
    return json.loads(result.stdout)


def _run_dom_actions(html: str, actions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Dispatch browser events against the actual generated catalog page."""
    program = r"""
const {JSDOM}=require('jsdom');
let raw='';process.stdin.on('data',x=>raw+=x);process.stdin.on('end',()=>{
 const input=JSON.parse(raw),dom=new JSDOM(input.html,{runScripts:'dangerously'}),d=dom.window.document;
 d.dispatchEvent(new dom.window.Event('DOMContentLoaded'));
 const event=(element,type)=>element.dispatchEvent(new dom.window.Event(type,{bubbles:true}));
 const compare=id=>[...d.querySelectorAll('[data-compare-select]')].find(box=>box.dataset.compareSelect===id);
 const control=name=>d.querySelector('[data-q='+name+']');
 const snapshot=label=>{const result=d.querySelector('[data-compare=result]');return {
   label, count:d.querySelector('[data-compare=count]').textContent,
   status:d.querySelector('[data-compare=status]').textContent,
   table:result.querySelectorAll('table').length,
   headers:[...result.querySelectorAll('th')].map(cell=>cell.textContent),
   headerScopes:[...result.querySelectorAll('th')].map(cell=>cell.scope),
   rows:[...result.querySelectorAll('tr')].map(row=>[...row.children].map(cell=>cell.textContent)),
   text:result.textContent,
   links:[...result.querySelectorAll('a')].map(link=>({href:link.getAttribute('href'),text:link.textContent})),
   visible:[...d.querySelectorAll('[data-model-id]')].filter(cell=>!cell.parentElement.hidden).map(cell=>cell.dataset.modelId),
   selected:[...d.querySelectorAll('[data-compare-select]:checked')].map(box=>box.dataset.compareSelect),
   active:d.activeElement===d.body?'body':(d.activeElement.dataset.compare||d.activeElement.dataset.compareSelect||d.activeElement.dataset.q||d.activeElement.tagName),
   live:d.querySelector('[data-compare=status]').getAttribute('aria-live'),
   details:[...result.querySelectorAll('details')].map(item=>item.querySelector('summary')?.textContent),
   labels:[...d.querySelectorAll('[data-compare-select]')].map(box=>box.labels?.[0]?.textContent),
   controlTypes:[...d.querySelectorAll('[data-compare-select]')].map(box=>box.type),
   scripts:result.querySelectorAll('script').length,
   index:JSON.parse(d.querySelector('#model-query-index').textContent).map(item=>item.samyak_id),
 };};
 const out=[];
 for(const action of input.actions){
   if(action.type==='select'||action.type==='remove'){
     const box=compare(action.id);if(!box)throw new Error('missing checkbox '+action.id);box.focus();box.checked=action.type==='select';event(box,'change');
   }else if(action.type==='clear'){const button=d.querySelector('[data-compare=clear]');button.focus();button.click();
   }else if(action.type==='compare_doc'){const box=[...d.querySelectorAll('[data-compare-doc]')].find(item=>item.value===action.value);box.focus();box.checked=action.checked;event(box,'change');
   }else if(action.type==='legacy'){const box=d.querySelector('[data-compare=legacy]');box.focus();box.checked=action.checked;event(box,'change');
   }else if(action.type==='filter_select'){
     const element=control(action.name);element.focus();for(const option of element.options)option.selected=action.values.includes(option.value);event(element,'input');
   }else if(action.type==='filter_value'){
     const element=control(action.name);element.focus();element.value=action.value;event(element,'input');
   }else if(action.type==='focus'){const element=action.selector==='clear'?d.querySelector('[data-compare=clear]'):control(action.selector);element.focus();
   }else throw new Error('unknown action '+action.type);
   out.push(snapshot(action.label));
 }
 process.stdout.write(JSON.stringify(out));
});"""
    env = {**os.environ, "NODE_PATH": "/private/tmp/samyak-dom-tests/node_modules"}
    result = subprocess.run(
        ["node", "-e", program],
        input=json.dumps({"html": html, "actions": actions}),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    return json.loads(result.stdout)


def _states(html: str, actions: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {item["label"]: item for item in _run_dom_actions(html, actions)}


def _provider_model(provider: str, provider_model_id: str, **fields: object):
    record = _model(provider_model_id, **fields)
    return replace(
        record,
        identity=replace(record.identity, provider_id=provider),
        evidence_disposition=EvidenceDisposition.CURRENT_DOCUMENTED,
    )


def test_comparison_workspace_dom_selection_clear_order_and_limit() -> None:
    catalog = _sample_catalog()
    fourth = _model(
        "gpt-fourth",
        lifecycle=_known(LifecycleState.ACTIVE),
        api_access=_known(True),
    )
    catalog = _catalog(
        tuple(sorted(catalog.models + (fourth,), key=lambda item: item.samyak_id)),
        notices=catalog.notices,
    )
    html = render_model_catalog(catalog)
    actual = _run_dom(html)
    states = {item["label"]: item for item in actual["out"]}
    assert actual["scripts"] is True
    assert states["empty"]["count"] == "0 selected"
    assert states["one"]["tables"] == 0
    assert states["two"]["tables"] == 1
    assert states["two"]["headers"][1:3] == ["openai:gpt-active", "openai:gpt-conflict"]
    assert states["three"]["count"] == "3 selected"
    assert states["four"]["count"] == "3 selected"
    assert "Only three models" in states["four"]["status"]
    assert states["clear"]["count"] == "0 selected"
    assert states["clear"]["tables"] == 0
    assert all(label.startswith(" Compare openai:") for label in actual["labels"])


def test_comparison_workspace_renders_conflict_claim_provenance() -> None:
    """Competing claims render as text with their individual provenance."""
    catalog = _sample_catalog()
    conflict = next(model for model in catalog.models if model.provider_model_id == "gpt-conflict")
    catalog = _catalog(
        tuple(
            replace(
                model,
                display_name=_conflict('<script>alert("x")</script>', "<b>alternate</b>"),
            )
            if model is conflict
            else model
            for model in catalog.models
        ),
        notices=catalog.notices,
    )
    html = render_model_catalog(catalog)
    actual = _run_dom(html)
    states = {item["label"]: item for item in actual["out"]}
    rendered = states["three"]
    text = rendered["text"]

    assert "Provider" in text
    assert "Documentation disposition" in text
    assert "Stable model ID" in text
    assert "Display name" in text
    assert "openai" in text
    assert "current_documented" in text
    assert "openai:gpt-active" in text
    assert '<script>alert("x")</script>' in text
    assert "<b>alternate</b>" in text
    assert "Claim value: true; source ID: source-0; verified: 2026-09-10T17:35:13+00:00" in text
    assert "Claim value: false; source ID: source-1; verified: 2026-09-10T17:35:13+00:00" in text
    assert 'Claim value: <script>alert("x")</script>; source ID: source-0' in text
    assert "Claim value: <b>alternate</b>; source ID: source-1" in text
    assert rendered["scripts"] == 0
    assert rendered["links"] == [
        {"href": "/models/openai:gpt-active", "text": "Open model evidence details"},
        {"href": "/models/openai:gpt-conflict", "text": "Open model evidence details"},
        {"href": "/models/openai:gpt-retired", "text": "Open model evidence details"},
    ]


def test_dom_selection_removal_and_filters_preserve_the_compare_set() -> None:
    """Filters hide catalog rows only; they never reset comparison selection."""
    alpha = _provider_model(
        "openai",
        "alpha",
        display_name=_known("Alpha"),
        lifecycle=_known(LifecycleState.ACTIVE),
        api_access=_known(True),
        input_modalities=_known((Modality.TEXT,)),
        context_window=_known(ContextWindow(128, ContextWindowKind.INPUT)),
    )
    beta = _provider_model(
        "google",
        "beta",
        display_name=_known("Beta"),
        lifecycle=_known(LifecycleState.ACTIVE),
        api_access=_known(True),
        input_modalities=_known((Modality.TEXT, Modality.IMAGE)),
        context_window=_known(ContextWindow(256, ContextWindowKind.INPUT)),
    )
    gamma = _provider_model(
        "anthropic",
        "gamma",
        display_name=_known("Gamma"),
        lifecycle=_known(LifecycleState.ACTIVE),
        api_access=_known(False),
        input_modalities=_known((Modality.TEXT,)),
        context_window=_known(ContextWindow(64, ContextWindowKind.INPUT)),
    )
    catalog = _catalog((alpha, beta, gamma))
    alpha_id, beta_id, gamma_id = (alpha.samyak_id, beta.samyak_id, gamma.samyak_id)
    states = _states(
        render_model_catalog(catalog),
        [
            {"type": "select", "id": alpha_id, "label": "first"},
            {"type": "select", "id": beta_id, "label": "second"},
            {"type": "select", "id": gamma_id, "label": "third"},
            {"type": "remove", "id": alpha_id, "label": "remove-first"},
            {"type": "remove", "id": beta_id, "label": "remove-middle"},
            {"type": "clear", "label": "clear"},
            {"type": "select", "id": alpha_id, "label": "first-again"},
            {"type": "select", "id": beta_id, "label": "second-again"},
            {"type": "select", "id": gamma_id, "label": "third-again"},
            {"type": "remove", "id": gamma_id, "label": "remove-last"},
            {
                "type": "filter_select",
                "name": "provider",
                "values": ["google"],
                "label": "provider-filter",
            },
            {"type": "filter_value", "name": "api", "value": "true", "label": "api-filter"},
            {
                "type": "filter_select",
                "name": "input",
                "values": ["image"],
                "label": "modality-filter",
            },
            {"type": "filter_value", "name": "context", "value": "100", "label": "context-filter"},
            {"type": "filter_value", "name": "kind", "value": "input", "label": "kind-filter"},
        ],
    )

    assert states["third"]["headers"][1:4] == [alpha_id, beta_id, gamma_id]
    assert states["remove-first"]["headers"][1:3] == [beta_id, gamma_id]
    assert states["remove-first"]["active"] == alpha_id
    assert states["remove-middle"]["table"] == 0
    assert states["remove-middle"]["status"] == "Select two or three models to compare."
    assert states["clear"]["active"] == "clear"
    assert states["clear"]["table"] == 0
    assert states["remove-last"]["headers"][1:3] == [alpha_id, beta_id]
    for state in (
        "provider-filter",
        "api-filter",
        "modality-filter",
        "context-filter",
        "kind-filter",
    ):
        assert set(states[state]["selected"]) == {alpha_id, beta_id}
        assert states[state]["headers"][1:3] == [alpha_id, beta_id]
    assert states["provider-filter"]["visible"] == [beta_id]
    assert states["kind-filter"]["active"] == "kind"


def test_dom_documentation_eligibility_opt_ins_and_mixed_warning() -> None:
    current = replace(
        _model("current", display_name=_known("Current")),
        evidence_disposition=EvidenceDisposition.CURRENT_DOCUMENTED,
    )
    historical = replace(
        _model("historical", display_name=_known("Historical")),
        evidence_disposition=EvidenceDisposition.HISTORICAL_REFERENCE_ONLY,
    )
    legacy = _model("legacy", display_name=_known("Legacy"))
    catalog = _catalog((current, historical, legacy))
    catalog = replace(
        catalog,
        models=tuple(
            replace(model, evidence_disposition=None)
            if model.samyak_id == legacy.samyak_id
            else model
            for model in catalog.models
        ),
    )
    states = _states(
        render_model_catalog(catalog),
        [
            {"type": "select", "id": current.samyak_id, "label": "current"},
            {"type": "select", "id": historical.samyak_id, "label": "strict-historical"},
            {
                "type": "compare_doc",
                "value": "historical_reference_only",
                "checked": True,
                "label": "historical-on",
            },
            {
                "type": "compare_doc",
                "value": "historical_reference_only",
                "checked": False,
                "label": "historical-off",
            },
            {"type": "remove", "id": historical.samyak_id, "label": "remove-historical"},
            {"type": "legacy", "checked": True, "label": "legacy-on"},
            {"type": "select", "id": legacy.samyak_id, "label": "legacy-selected"},
            {"type": "legacy", "checked": False, "label": "legacy-off"},
        ],
    )

    assert "historical_reference_only; opt in explicitly" in states["strict-historical"]["status"]
    assert states["strict-historical"]["table"] == 0
    assert states["historical-on"]["table"] == 1
    assert "multiple documentation certification levels" in states["historical-on"]["text"]
    assert historical.samyak_id in states["historical-off"]["selected"]
    assert states["historical-off"]["table"] == 0
    assert states["legacy-selected"]["table"] == 1
    assert "legacy_uncertified" in states["legacy-selected"]["text"]
    assert "multiple documentation certification levels" in states["legacy-selected"]["text"]
    assert states["legacy-off"]["table"] == 0
    assert "legacy_uncertified; opt in explicitly" in states["legacy-off"]["status"]


def test_dom_embedded_server_scope_cannot_expand_after_provider_restriction() -> None:
    openai = _provider_model("openai", "scoped-openai", display_name=_known("Scoped OpenAI"))
    google = _provider_model("google", "scoped-google", display_name=_known("Scoped Google"))
    catalog = _catalog((openai, google))
    html = render_model_catalog(catalog, provider="openai")
    html = html.replace(
        "</body>",
        '<label><input type="checkbox" data-compare-select="google:scoped-google"> Attempt expansion</label></body>',
    )
    states = _states(
        html,
        [
            {"type": "select", "id": openai.samyak_id, "label": "scoped"},
            {"type": "select", "id": google.samyak_id, "label": "out-of-scope"},
        ],
    )

    assert states["scoped"]["index"] == [openai.samyak_id]
    assert states["scoped"]["visible"] == [openai.samyak_id]
    assert states["out-of-scope"]["table"] == 0
    assert "unknown comparison model ID: google:scoped-google" in states["out-of-scope"]["status"]
    assert google.samyak_id not in states["out-of-scope"]["index"]


def test_real_catalog_dom_trios_preserve_evidence_and_comparison_boundaries() -> None:
    """Exercise real captured catalog records through generated HTML and events."""
    configured_path = os.environ.get("SAMYAK_REAL_CATALOG_PATH")
    if not configured_path:
        pytest.skip("set SAMYAK_REAL_CATALOG_PATH to an accepted models/catalog.json")
    catalog_path = Path(configured_path).expanduser()
    if not catalog_path.is_file() or catalog_path.name != "catalog.json":
        pytest.skip("SAMYAK_REAL_CATALOG_PATH must name an accepted models/catalog.json")
    catalog = FileCatalogStore(root=catalog_path.parent.parent).load()
    expected_ids = {
        "anthropic:claude-opus-4-6",
        "anthropic:claude-fable-5",
        "google:gemini-2.5-pro",
        "google:gemini-2.5-flash",
        "google:gemini-2.5-flash-lite",
        "fireworks:accounts/fireworks/models/kimi-k2p5",
        "fireworks:accounts/fireworks/models/kimi-k3",
    }
    available = {record.samyak_id for record in catalog.models}
    if not expected_ids <= available:
        pytest.skip("accepted real-catalog DOM examples are unavailable")
    enterprise = [
        "anthropic:claude-opus-4-6",
        "google:gemini-2.5-pro",
        "fireworks:accounts/fireworks/models/kimi-k2p5",
    ]
    cross_provider = [
        "anthropic:claude-fable-5",
        "google:gemini-2.5-flash",
        "fireworks:accounts/fireworks/models/kimi-k3",
    ]
    multimodal = [
        "google:gemini-2.5-pro",
        "google:gemini-2.5-flash",
        "google:gemini-2.5-flash-lite",
    ]
    rendered_trios: dict[str, dict[str, object]] = {}
    for name, trio in (
        ("enterprise", enterprise),
        ("cross", cross_provider),
        ("multimodal", multimodal),
    ):
        records = tuple(record for record in catalog.models if record.samyak_id in trio)
        states = _states(
            render_model_catalog(_catalog(records)),
            [
                {"type": "select", "id": samyak_id, "label": f"{name}-{index}"}
                for index, samyak_id in enumerate(trio)
            ],
        )
        rendered = states[f"{name}-2"]
        rendered_trios[name] = rendered
        assert rendered["headers"][1:4] == trio
        assert "Provider" in rendered["text"]
        assert "Documentation disposition" in rendered["text"]
        assert "current_documented" in rendered["text"]
        assert "Sources:" in rendered["text"]
        assert "verified:" in rendered["text"]
        assert rendered["links"] == [
            {"href": model_href(samyak_id), "text": "Open model evidence details"}
            for samyak_id in trio
        ]

    enterprise_text = rendered_trios["enterprise"]["text"]
    assert '"kind":"combined"' in enterprise_text
    assert '"kind":"unknown"' in enterprise_text
    assert "not all selected records have context evidence" in enterprise_text
    assert "availability scope is not verified for all" in enterprise_text
    multimodal_text = rendered_trios["multimodal"]["text"]
    assert "image" in multimodal_text and "video" in multimodal_text and "audio" in multimodal_text
    assert '"kind":"combined"' in multimodal_text
    assert "context_window: not directly comparable" not in multimodal_text


def test_dom_accessibility_labels_live_regions_focus_and_disclosures() -> None:
    catalog = _sample_catalog()
    ids = [model.samyak_id for model in catalog.models]
    states = _states(
        render_model_catalog(catalog),
        [
            {"type": "select", "id": ids[0], "label": "one"},
            {"type": "select", "id": ids[1], "label": "two"},
            {"type": "remove", "id": ids[1], "label": "remove"},
            {"type": "select", "id": ids[1], "label": "two-again"},
            {"type": "focus", "selector": "provider", "label": "provider-focus"},
            {
                "type": "filter_select",
                "name": "provider",
                "values": ["openai"],
                "label": "provider-change",
            },
            {"type": "clear", "label": "clear"},
        ],
    )

    rendered = states["two"]
    assert all(label and label.startswith(" Compare openai:") for label in rendered["labels"])
    assert set(rendered["controlTypes"]) == {"checkbox"}
    assert rendered["live"] == "polite"
    assert "col" in rendered["headerScopes"] and "row" in rendered["headerScopes"]
    assert rendered["details"]
    assert set(rendered["details"]) == {"Evidence"}
    assert states["remove"]["active"] == ids[1]
    assert states["provider-change"]["active"] == "provider"
    assert states["clear"]["active"] == "clear"
    assert states["clear"]["table"] == 0
