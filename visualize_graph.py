import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

NODE_COLORS = {"EVENT": "#1f77b4", "SESSION": "#ff7f0e", "EPISODE": "#2ca02c", "NARRATIVE": "#9467bd", "ENTITY": "#8c564b"}
EDGE_COLORS = {"TEMPORAL": "#d62728", "SEMANTIC": "#17becf", "CAUSAL": "#bcbd22", "ENTITY": "#7f7f7f"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize graph.json as PNG and/or interactive HTML.")
    p.add_argument("--graph", required=True)
    p.add_argument("--output", help="PNG output path")
    p.add_argument("--html-output", help="HTML output path")
    p.add_argument("--max-nodes", type=int, default=80)
    p.add_argument("--center-node")
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--node-types")
    p.add_argument("--link-types")
    p.add_argument("--layout", choices=["spring", "kamada", "shell", "spectral"], default="spring")
    p.add_argument("--label-mode", choices=["none", "type", "short"], default="type")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def norm(raw: Optional[str]) -> Optional[Set[str]]:
    return {x.strip().upper() for x in raw.split(",") if x.strip()} if raw else None


def load_payload(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(payload: Dict, node_types: Optional[Set[str]], link_types: Optional[Set[str]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node in payload.get("nodes", []):
        node_type = str(node.get("node_type", "UNKNOWN")).upper()
        if node_types and node_type not in node_types:
            continue
        graph.add_node(node["node_id"], node_type=node_type, content=node.get("content_narrative") or node.get("summary") or "", timestamp=node.get("timestamp") or node.get("date_time") or "")
    for link in payload.get("links", []):
        src = link.get("source_node_id")
        dst = link.get("target_node_id")
        link_type = str(link.get("link_type", "UNKNOWN")).upper()
        if link_types and link_type not in link_types:
            continue
        if src in graph and dst in graph:
            graph.add_edge(src, dst, key=link.get("link_id"), link_type=link_type, sub_type=link.get("properties", {}).get("sub_type", ""))
    return graph


def choose_subgraph(graph: nx.MultiDiGraph, max_nodes: int, center_node: Optional[str], hops: int) -> nx.MultiDiGraph:
    if center_node:
        if center_node not in graph:
            raise ValueError(f"Center node not found: {center_node}")
        visited, q, g2 = {center_node}, deque([(center_node, 0)]), graph.to_undirected()
        while q:
            current, depth = q.popleft()
            if depth >= hops:
                continue
            for neighbor in sorted(g2.neighbors(current), key=lambda n: g2.degree(n), reverse=True):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if max_nodes and len(visited) >= max_nodes:
                    break
                q.append((neighbor, depth + 1))
        return graph.subgraph(visited).copy()
    if max_nodes <= 0 or graph.number_of_nodes() <= max_nodes:
        return graph.copy()
    top = {node_id for node_id, _ in sorted(graph.degree, key=lambda item: item[1], reverse=True)[:max_nodes]}
    return graph.subgraph(top).copy()


def labels_for(graph: nx.MultiDiGraph, mode: str) -> Dict[str, str]:
    if mode == "none":
        return {}
    out = {}
    for node_id, data in graph.nodes(data=True):
        if mode == "type":
            out[node_id] = f"{data.get('node_type', 'NOD')[:3]}:{node_id[:6]}"
        else:
            text = str(data.get("content", "")).strip().replace("\n", " ")
            out[node_id] = (text[:24] + ("..." if len(text) > 24 else "")) or f"{data.get('node_type', 'NOD')[:3]}:{node_id[:6]}"
    return out


def layout_for(graph: nx.MultiDiGraph, layout: str, seed: int) -> Dict[str, Tuple[float, float]]:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    simple.add_edges_from((u, v) for u, v, _ in graph.edges(keys=True))
    if layout == "kamada":
        return nx.kamada_kawai_layout(simple)
    if layout == "shell":
        return nx.shell_layout(simple)
    if layout == "spectral":
        return nx.spectral_layout(simple)
    return nx.spring_layout(simple, seed=seed)


def draw_png(graph: nx.MultiDiGraph, pos: Dict[str, Tuple[float, float]], out: Path, label_mode: str, source: Path) -> None:
    labels = labels_for(graph, label_mode)
    fig, ax = plt.subplots(figsize=(16, 12), dpi=180)
    groups: Dict[str, List[str]] = {}
    for node_id, data in graph.nodes(data=True):
        groups.setdefault(data.get("node_type", "UNKNOWN"), []).append(node_id)
    for node_type, node_ids in sorted(groups.items()):
        nx.draw_networkx_nodes(graph, pos, nodelist=node_ids, node_color=NODE_COLORS.get(node_type, "#ccc"), node_size=260 if node_type == "EVENT" else 420, alpha=0.92, linewidths=0.8, edgecolors="#fff", ax=ax)
    edge_groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for src, dst, key, data in graph.edges(keys=True, data=True):
        edge_groups.setdefault(data.get("link_type", "UNKNOWN"), []).append((src, dst, key))
    for link_type, edge_triplets in sorted(edge_groups.items()):
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _ in edge_triplets], edge_color=EDGE_COLORS.get(link_type, "#999"), alpha=0.18, width=0.9 if link_type == "TEMPORAL" else 0.7, arrows=False, ax=ax)
    if labels:
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, font_color="#1f2937", ax=ax)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", label=f"{t} ({len(ids)})", markerfacecolor=NODE_COLORS.get(t, "#ccc"), markersize=9) for t, ids in sorted(groups.items())] + [Line2D([0], [0], color=EDGE_COLORS.get(t, "#999"), label=f"{t} ({len(es)})", linewidth=2) for t, es in sorted(edge_groups.items())], loc="upper left", frameon=True, framealpha=0.95, fontsize=9)
    node_counts = Counter(data.get("node_type", "UNKNOWN") for _, data in graph.nodes(data=True))
    edge_counts = Counter(data.get("link_type", "UNKNOWN") for _, _, _, data in graph.edges(keys=True, data=True))
    ax.set_title("MAGMA Memory Graph", fontsize=18, weight="bold", pad=16)
    fig.text(0.5, 0.02, f"Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}  Node types: {dict(node_counts)}  Link types: {dict(edge_counts)}\nSource: {source}", ha="center", fontsize=9)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def html_payload(graph: nx.MultiDiGraph, pos: Dict[str, Tuple[float, float]], label_mode: str) -> Dict:
    labels = labels_for(graph, label_mode)
    nodes = [{"id": n, "nodeType": d.get("node_type", "UNKNOWN"), "content": d.get("content", ""), "timestamp": d.get("timestamp", ""), "label": labels.get(n, ""), "x": float(pos[n][0]), "y": float(pos[n][1])} for n, d in graph.nodes(data=True)]
    edges = [{"id": key, "source": src, "target": dst, "linkType": data.get("link_type", "UNKNOWN"), "subType": data.get("sub_type", "")} for src, dst, key, data in graph.edges(keys=True, data=True)]
    return {"nodes": nodes, "edges": edges, "nodeColors": NODE_COLORS, "edgeColors": EDGE_COLORS}


def render_html(graph: nx.MultiDiGraph, pos: Dict[str, Tuple[float, float]], out: Path, label_mode: str, source: Path) -> None:
    payload = json.dumps(html_payload(graph, pos, label_mode), ensure_ascii=False)
    node_counts = json.dumps(dict(Counter(data.get("node_type", "UNKNOWN") for _, data in graph.nodes(data=True))), ensure_ascii=False)
    edge_counts = json.dumps(dict(Counter(data.get("link_type", "UNKNOWN") for _, _, _, data in graph.edges(keys=True, data=True))), ensure_ascii=False)
    template = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MAGMA Graph Viewer</title><style>
body{margin:0;font-family:Segoe UI,Noto Sans,sans-serif;color:#1f2937;background:linear-gradient(180deg,#fcfbf7,#f4f1ea)}.page{display:grid;grid-template-columns:320px 1fr;min-height:100vh}.side{padding:20px;background:rgba(255,252,247,.94);border-right:1px solid rgba(31,41,55,.12);overflow:auto}.panel{margin:0 0 16px;padding:14px;border:1px solid rgba(31,41,55,.12);border-radius:16px;background:rgba(255,255,255,.7)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.stat strong{display:block;font-size:1.1rem}.muted{color:#6b7280}.checklist{display:grid;gap:8px}.checklist label{display:flex;gap:10px;align-items:center}.sw{width:10px;height:10px;border-radius:999px;display:inline-block;border:1px solid rgba(0,0,0,.12)}input[type=search]{width:100%;padding:10px 12px;border-radius:12px;border:1px solid rgba(31,41,55,.12)}.viewer{position:relative;overflow:hidden}#canvas{width:100%;height:100vh;display:block;cursor:grab}#canvas.dragging{cursor:grabbing}.tip{position:fixed;pointer-events:none;max-width:320px;padding:12px 14px;border-radius:14px;background:rgba(17,24,39,.92);color:#f9fafb;font-size:.85rem;line-height:1.5;opacity:0;transition:opacity .12s ease;z-index:10}.tip.show{opacity:1}.bar{position:absolute;top:18px;right:18px;display:flex;gap:10px;z-index:5}.bar button{border:0;border-radius:999px;padding:10px 14px;background:rgba(255,252,247,.92);cursor:pointer}code{font-family:Consolas,Courier New,monospace;font-size:.82rem;background:rgba(15,23,42,.06);padding:1px 5px;border-radius:6px}@media (max-width:980px){.page{grid-template-columns:1fr}#canvas{height:70vh}}</style></head><body>
<div class="page"><aside class="side"><h1 style="margin:0 0 8px">MAGMA Graph Viewer</h1><p class="muted">Zoom, pan, drag nodes, filter types, and inspect node details.</p><div class="panel grid"><div class="stat"><strong>__NODES__</strong><span>Rendered nodes</span></div><div class="stat"><strong>__EDGES__</strong><span>Rendered edges</span></div></div><div class="panel"><strong>Source</strong><div style="margin-top:8px"><code>__SOURCE__</code></div></div><div class="panel"><strong>Search</strong><div style="margin-top:8px"><input id="searchInput" type="search" placeholder="Search node id, content, or type"></div></div><div class="panel"><strong>Node Types</strong><div id="nodeFilters" class="checklist" style="margin-top:10px"></div></div><div class="panel"><strong>Link Types</strong><div id="edgeFilters" class="checklist" style="margin-top:10px"></div></div><div class="panel"><strong>Selected Node</strong><div id="selectedNode" class="muted" style="margin-top:8px">Click a node to pin details here.</div></div><div class="panel"><strong>Hints</strong><div class="muted" style="margin-top:8px">Wheel to zoom.<br>Drag background to pan.<br>Drag a node to reposition it.<br>Double-click empty space to reset view.</div></div></aside><main class="viewer"><div class="bar"><button id="fitBtn" type="button">Fit View</button><button id="resetBtn" type="button">Reset Filters</button></div><svg id="canvas" viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid meet"><g id="viewport"><g id="edges"></g><g id="nodes"></g><g id="labels"></g></g></svg></main></div><div id="tooltip" class="tip"></div>
<script>
const data=__PAYLOAD__, nodeCounts=__NODE_COUNTS__, edgeCounts=__EDGE_COUNTS__;
const svg=document.getElementById("canvas"), viewport=document.getElementById("viewport"), edgeLayer=document.getElementById("edges"), nodeLayer=document.getElementById("nodes"), labelLayer=document.getElementById("labels"), tooltip=document.getElementById("tooltip"), selectedNodeEl=document.getElementById("selectedNode"), searchInput=document.getElementById("searchInput"), nodeFilters=document.getElementById("nodeFilters"), edgeFilters=document.getElementById("edgeFilters");
const state={scale:1,tx:0,ty:0,dragCanvas:false,dragNode:null,dragX:0,dragY:0,search:"",selected:null,nodeTypes:new Set(Object.keys(nodeCounts)),edgeTypes:new Set(Object.keys(edgeCounts))};
const nodes=data.nodes.map(n=>({...n,visible:true})), nodeById=new Map(nodes.map(n=>[n.id,n])), edges=data.edges.map(e=>({...e,visible:true}));
function esc(t){return String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function fitCoords(){const xs=nodes.map(n=>n.x), ys=nodes.map(n=>n.y), minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys), spanX=Math.max(maxX-minX,.01), spanY=Math.max(maxY-minY,.01); nodes.forEach(n=>{n.sx=((n.x-minX)/spanX)*1200+200; n.sy=((n.y-minY)/spanY)*700+150;});}
function apply(){viewport.setAttribute("transform",`translate(${state.tx} ${state.ty}) scale(${state.scale})`);} function resetCamera(){state.scale=1;state.tx=0;state.ty=0;apply();}
function svgPointFromClient(clientX,clientY){const pt=svg.createSVGPoint(); pt.x=clientX; pt.y=clientY; return pt.matrixTransform(svg.getScreenCTM().inverse());}
function visibleDegree(id){let c=0; for(const e of edges) if(e.visible&&(e.source===id||e.target===id)) c++; return c;} function nodeOpacity(node){if(!state.selected) return .94; if(node.id===state.selected) return 1; for(const e of edges) if(e.visible&&((e.source===state.selected&&e.target===node.id)||(e.target===state.selected&&e.source===node.id))) return .96; return .14;} function edgeOpacity(edge){if(!state.selected) return .24; return (edge.source===state.selected||edge.target===state.selected) ? .88 : .05;} function labelOpacity(node){if(!node.label) return 0; if(state.scale>=1.3) return nodeOpacity(node); return state.selected===node.id?1:0;}
function updateVisibility(){const q=state.search.trim().toLowerCase(); nodes.forEach(n=>{const okType=state.nodeTypes.has(n.nodeType); const text=`${n.id} ${n.nodeType} ${n.content} ${n.timestamp}`.toLowerCase(); n.visible=okType&&(!q||text.includes(q));}); edges.forEach(e=>{const s=nodeById.get(e.source), t=nodeById.get(e.target); e.visible=state.edgeTypes.has(e.linkType)&&s&&t&&s.visible&&t.visible;});}
function showTip(evt,html){tooltip.innerHTML=html; tooltip.style.left=`${evt.clientX+14}px`; tooltip.style.top=`${evt.clientY+14}px`; tooltip.classList.add("show");} function hideTip(){tooltip.classList.remove("show");}
function setSelected(node){state.selected=node?node.id:null; if(!node){selectedNodeEl.innerHTML='<span class="muted">Click a node to pin details here.</span>'; render(); return;} selectedNodeEl.innerHTML=`<strong>${esc(node.label||node.id.slice(0,8))}</strong><br><code>${esc(node.id)}</code><br>Type: ${esc(node.nodeType)}<br>Timestamp: ${esc(node.timestamp||"-")}<br>Visible degree: ${visibleDegree(node.id)}<br><br>${esc(node.content||"(no content)")}`; render();}
function makeFilter(container,name,color,count,setRef){const label=document.createElement("label"), input=document.createElement("input"), sw=document.createElement("span"), txt=document.createElement("span"); input.type="checkbox"; input.checked=true; input.addEventListener("change",()=>{input.checked?setRef.add(name):setRef.delete(name); updateVisibility(); render();}); sw.className="sw"; sw.style.background=color; txt.textContent=`${name} (${count})`; label.append(input,sw,txt); container.appendChild(label);}
function render(){edgeLayer.innerHTML=""; nodeLayer.innerHTML=""; labelLayer.innerHTML=""; for(const e of edges){if(!e.visible) continue; const s=nodeById.get(e.source), t=nodeById.get(e.target), line=document.createElementNS("http://www.w3.org/2000/svg","line"); line.setAttribute("x1",s.sx); line.setAttribute("y1",s.sy); line.setAttribute("x2",t.sx); line.setAttribute("y2",t.sy); line.setAttribute("stroke",data.edgeColors[e.linkType]||"#999"); line.setAttribute("stroke-width",e.linkType==="TEMPORAL"?"1.4":"1.1"); line.setAttribute("stroke-opacity",edgeOpacity(e)); line.addEventListener("mousemove",evt=>showTip(evt,`<strong>${esc(e.linkType)}</strong><br>${esc(e.subType||"-")}<br><code>${esc(e.source.slice(0,8))}</code> -> <code>${esc(e.target.slice(0,8))}</code>`)); line.addEventListener("mouseleave",hideTip); edgeLayer.appendChild(line);} for(const n of nodes){if(!n.visible) continue; const r=n.nodeType==="EVENT"?8:12, c=document.createElementNS("http://www.w3.org/2000/svg","circle"); c.setAttribute("cx",n.sx); c.setAttribute("cy",n.sy); c.setAttribute("r",r); c.setAttribute("fill",data.nodeColors[n.nodeType]||"#ccc"); c.setAttribute("fill-opacity",nodeOpacity(n)); c.setAttribute("stroke",n.id===state.selected?"#111827":"#fff"); c.setAttribute("stroke-width",n.id===state.selected?"2.4":"1.3"); c.addEventListener("mousedown",evt=>{evt.stopPropagation(); state.dragNode=n.id;}); c.addEventListener("click",evt=>{evt.stopPropagation(); setSelected(n);}); c.addEventListener("mousemove",evt=>showTip(evt,`<strong>${esc(n.nodeType)}</strong><br><code>${esc(n.id)}</code><br>Degree: ${visibleDegree(n.id)}<br><br>${esc((n.content||"(no content)").slice(0,220))}`)); c.addEventListener("mouseleave",hideTip); nodeLayer.appendChild(c); if(n.label){const t=document.createElementNS("http://www.w3.org/2000/svg","text"); t.setAttribute("x",n.sx+r+4); t.setAttribute("y",n.sy+3); t.setAttribute("font-size","12"); t.setAttribute("fill","#1f2937"); t.setAttribute("fill-opacity",labelOpacity(n)); t.textContent=n.label; labelLayer.appendChild(t);}}}
svg.addEventListener("wheel",evt=>{evt.preventDefault(); const p=svgPointFromClient(evt.clientX,evt.clientY); const nextScale=Math.min(8,Math.max(.25,state.scale*(evt.deltaY<0?1.1:.9))); const gx=(p.x-state.tx)/state.scale; const gy=(p.y-state.ty)/state.scale; state.scale=nextScale; state.tx=p.x-gx*state.scale; state.ty=p.y-gy*state.scale; apply(); render();},{passive:false}); svg.addEventListener("mousedown",evt=>{if(evt.target===svg){state.dragCanvas=true; svg.classList.add("dragging"); state.dragX=evt.clientX; state.dragY=evt.clientY;}});
window.addEventListener("mousemove",evt=>{if(state.dragCanvas){state.tx+=evt.clientX-state.dragX; state.ty+=evt.clientY-state.dragY; state.dragX=evt.clientX; state.dragY=evt.clientY; apply(); return;} if(state.dragNode){const n=nodeById.get(state.dragNode), p=svgPointFromClient(evt.clientX,evt.clientY); n.sx=(p.x-state.tx)/state.scale; n.sy=(p.y-state.ty)/state.scale; render();}}); window.addEventListener("mouseup",()=>{state.dragCanvas=false; state.dragNode=null; svg.classList.remove("dragging");});
svg.addEventListener("dblclick",evt=>{if(evt.target===svg) resetCamera();}); svg.addEventListener("click",evt=>{if(evt.target===svg) setSelected(null);}); searchInput.addEventListener("input",evt=>{state.search=evt.target.value; updateVisibility(); render();}); document.getElementById("fitBtn").addEventListener("click",resetCamera);
document.getElementById("resetBtn").addEventListener("click",()=>{state.nodeTypes=new Set(Object.keys(nodeCounts)); state.edgeTypes=new Set(Object.keys(edgeCounts)); state.search=""; searchInput.value=""; nodeFilters.querySelectorAll("input").forEach(i=>i.checked=true); edgeFilters.querySelectorAll("input").forEach(i=>i.checked=true); updateVisibility(); setSelected(null); resetCamera(); render();});
for(const [name,count] of Object.entries(nodeCounts)) makeFilter(nodeFilters,name,data.nodeColors[name]||"#ccc",count,state.nodeTypes); for(const [name,count] of Object.entries(edgeCounts)) makeFilter(edgeFilters,name,data.edgeColors[name]||"#999",count,state.edgeTypes); fitCoords(); updateVisibility(); apply(); render();
</script></body></html>"""
    html = template.replace("__PAYLOAD__", payload).replace("__NODE_COUNTS__", node_counts).replace("__EDGE_COUNTS__", edge_counts).replace("__NODES__", str(graph.number_of_nodes())).replace("__EDGES__", str(graph.number_of_edges())).replace("__SOURCE__", str(source))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph)
    png_path = Path(args.output) if args.output else graph_path.with_name("graph_viz.png")
    html_path = Path(args.html_output) if args.html_output else graph_path.with_name("graph_viz.html")
    graph = choose_subgraph(build_graph(load_payload(graph_path), norm(args.node_types), norm(args.link_types)), args.max_nodes, args.center_node, args.hops)
    pos = layout_for(graph, args.layout, args.seed)
    if args.output:
        draw_png(graph, pos, png_path, args.label_mode, graph_path)
        print(f"Saved graph visualization PNG to: {png_path}")
    if args.html_output or not args.output:
        render_html(graph, pos, html_path, args.label_mode, graph_path)
        print(f"Saved graph visualization HTML to: {html_path}")
    print(f"Rendered nodes: {graph.number_of_nodes()}")
    print(f"Rendered edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
