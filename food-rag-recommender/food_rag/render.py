"""HTML rendering for search results (kept separate from app.py so it can be unit-tested)."""

from __future__ import annotations

from html import escape
from typing import Sequence

from .search import SearchResult


def _card(r: SearchResult, rank: int) -> str:
    score = max(0.0, min(1.0, r["similarity_score"]))
    chips = "".join(f'<span class="chip">{escape(i)}</span>' for i in r["food_ingredients"][:6])
    macros = ""
    if r.get("protein_g") or r.get("carbs_g") or r.get("fat_g"):
        macros = (f'<div class="macros"><span>🥩 {r["protein_g"]:g}g protein</span>'
                  f'<span>🍞 {r["carbs_g"]:g}g carbs</span><span>🧈 {r["fat_g"]:g}g fat</span></div>')
    benefit = (f'<div class="benefit">💚 {escape(r["food_health_benefits"])}</div>'
               if r.get("food_health_benefits") else "")
    return f"""
<div class="food-card">
  <div class="card-head">
    <span class="rank">#{rank}</span>
    <h3>{escape(r['food_name'])}</h3>
  </div>
  <div class="meta">
    <span class="badge">{escape(r['cuisine_type'])}</span>
    <span class="badge kcal">🔥 {r['food_calories_per_serving']} kcal</span>
    {f'<span class="badge">{escape(r["cooking_method"])}</span>' if r.get("cooking_method") else ""}
  </div>
  <div class="match"><div class="bar" style="width:{score * 100:.0f}%"></div></div>
  <div class="match-label">{score * 100:.1f}% semantic match</div>
  <p class="desc">{escape(r['food_description'])}</p>
  {macros}
  {benefit}
  <div class="chips">{chips}</div>
</div>"""


def render_results(results: Sequence[SearchResult], empty_message: str = "No matching dishes found.") -> str:
    if not results:
        return (f'<div class="empty">🍽️ {escape(empty_message)}<br>'
                '<small>Try broader keywords or relax a filter.</small></div>')
    return '<div class="card-grid">' + "".join(_card(r, i) for i, r in enumerate(results, 1)) + "</div>"


CSS = """
.card-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }
.food-card { border:1px solid var(--border-color-primary); border-radius:14px; padding:14px 16px;
             background:var(--block-background-fill); }
.card-head { display:flex; align-items:baseline; gap:8px; }
.card-head h3 { margin:0; font-size:1.05rem; }
.rank { color:var(--body-text-color-subdued); font-weight:600; font-size:.85rem; }
.meta { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
.badge { font-size:.75rem; padding:2px 8px; border-radius:999px; background:var(--background-fill-secondary);
         border:1px solid var(--border-color-primary); }
.badge.kcal { font-weight:600; }
.match { height:6px; border-radius:6px; background:var(--background-fill-secondary); overflow:hidden; }
.match .bar { height:100%; background:linear-gradient(90deg,#f97316,#22c55e); }
.match-label { font-size:.72rem; color:var(--body-text-color-subdued); margin-top:3px; }
.desc { font-size:.88rem; margin:8px 0; line-height:1.4; }
.macros { display:flex; flex-wrap:wrap; gap:10px; font-size:.78rem; color:var(--body-text-color-subdued); }
.benefit { font-size:.8rem; margin-top:6px; }
.chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; }
.chip { font-size:.7rem; padding:1px 7px; border-radius:6px; background:var(--background-fill-secondary); }
.empty { text-align:center; padding:28px; color:var(--body-text-color-subdued); }
.status-line { font-size:.85rem; color:var(--body-text-color-subdued); }
"""
