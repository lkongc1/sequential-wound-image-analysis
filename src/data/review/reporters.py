"""Interactive HTML reporter for dataset review following SOLID principles.

SRP: Only handles HTML generation for the interactive viewer.
OCP: Easy to extend with additional viewer features.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InteractiveHTMLReporter:
    """Generates an interactive HTML viewer with embedded dataset.

    The viewer loads data from an embedded JSON array and provides:
    - Sortable/filterable table
    - Image modal viewer
    - Review status management with localStorage persistence
    - CSV export with updated review statuses

    Attributes:
        data_dir: Root directory for resolving relative image paths.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir

    def generate(
        self,
        csv_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> Path:
        """Generate interactive HTML viewer from CSV data.

        Args:
            csv_path: Path to the review CSV file.
            output_path: Path for the output HTML file.
            **kwargs: Additional options (data_dir for relative paths).

        Returns:
            Path to the generated HTML file.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        data_dir = kwargs.get("data_dir", csv_path.parent)
        df = pd.read_csv(csv_path)

        # Add _idx for tracking
        df["_idx"] = df.index

        # Ensure review_status column exists
        if "review_status" not in df.columns:
            df["review_status"] = ""

        # Ensure mask_edge_density exists (alias edge_density -> mask_edge_density)
        if "edge_density" in df.columns and "mask_edge_density" not in df.columns:
            df["mask_edge_density"] = df["edge_density"]

        # Convert DataFrame to list of dicts for JSON embedding
        records = df.to_dict(orient="records")

        # Make paths relative to data_dir for the viewer
        for record in records:
            for key in ["image_path", "mask_path"]:
                if key in record and record[key] and isinstance(record[key], str):
                    try:
                        p = Path(record[key])
                        if p.is_absolute():
                            record[key] = str(p.relative_to(data_dir))
                    except ValueError:
                        pass

        # FIX: Use pandas Series methods safely with type hints
        wound_min = 0.0
        wound_max = 100.0
        bright_min = 0
        bright_max = 255

        # FIX: Use pd.Series with explicit type handling for min/max
        if "wound_percentage" in df.columns:
            wound_series: pd.Series = df["wound_percentage"].dropna()
            if not wound_series.empty:
                wound_min = float(wound_series.min())  # type: ignore[arg-type]
                wound_max = float(wound_series.max())  # type: ignore[arg-type]

        if "brightness_mean" in df.columns:
            bright_series: pd.Series = df["brightness_mean"].dropna()
            if not bright_series.empty:
                bright_min = int(bright_series.min())  # type: ignore[arg-type]
                bright_max = int(bright_series.max())  # type: ignore[arg-type]

        html = self._generate_html(
            records,
            str(csv_path.name),
            wound_min=wound_min,
            wound_max=wound_max,
            bright_min=bright_min,
            bright_max=bright_max,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.info(f"Interactive viewer generated: {output_path}")
        return output_path

    def _generate_html(
        self,
        records: List[Dict],
        csv_filename: str,
        wound_min: float = 0.0,
        wound_max: float = 100.0,
        bright_min: int = 0,
        bright_max: int = 255,
    ) -> str:
        """Generate complete HTML with embedded data."""
        data_json = json.dumps(records, ensure_ascii=False)

        wound_min_int = int(wound_min)
        wound_max_int = int(wound_max)
        bright_min_int = int(bright_min)
        bright_max_int = int(bright_max)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wound Dataset Review</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #f59e0b;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; padding: 16px; }}

        header {{
            background: linear-gradient(135deg, var(--primary) 0%, #1d4ed8 100%);
            color: white;
            padding: 20px 24px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        header h1 {{ font-size: 1.4rem; font-weight: 600; }}
        header .subtitle {{ opacity: 0.9; font-size: 0.85rem; margin-top: 4px; }}

        .stats-bar {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}
        .stat-pill {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .stat-pill .value {{ font-weight: 700; color: var(--primary); }}
        .stat-pill.danger .value {{ color: var(--danger); }}
        .stat-pill.success .value {{ color: var(--success); }}
        .stat-pill.warning .value {{ color: var(--warning); }}

        .filters-panel {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .filters-row {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filters-row input, .filters-row select {{
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.85rem;
        }}
        .filters-row input[type="text"] {{ flex: 1; min-width: 180px; }}
        .filters-row label {{ font-size: 0.8rem; display: flex; align-items: center; gap: 4px; cursor: pointer; }}

        .range-filter {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .range-filter label {{ white-space: nowrap; }}
        .range-values {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 0.8rem;
            min-width: 80px;
        }}
        .range-values input[type="number"] {{
            width: 60px;
            padding: 4px 6px;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 0.8rem;
            text-align: center;
        }}

        .dual-range {{
            position: relative;
            height: 20px;
            width: 140px;
        }}
        .dual-range input[type="range"] {{
            position: absolute;
            width: 100%;
            pointer-events: none;
            -webkit-appearance: none;
            background: none;
            top: 0;
            height: 100%;
        }}
        .dual-range input[type="range"]::-webkit-slider-thumb {{
            pointer-events: auto;
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            margin-top: -6px;
        }}
        .dual-range input[type="range"]::-moz-range-thumb {{
            pointer-events: auto;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            border: none;
        }}
        .dual-range input[type="range"]::-webkit-slider-runnable-track {{
            height: 2px;
            background: var(--border);
            border-radius: 1px;
        }}
        .dual-range input[type="range"]::-moz-range-track {{
            height: 2px;
            background: var(--border);
            border-radius: 1px;
        }}
        .dual-range input[type="range"]:focus {{ outline: none; }}

        .actions-bar {{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .btn-primary {{ background: var(--primary); color: white; }}
        .btn-primary:hover {{ background: #1d4ed8; }}
        .btn-secondary {{ background: var(--border); color: var(--text); }}
        .btn-secondary:hover {{ background: #cbd5e1; }}
        .btn-danger {{ background: var(--danger); color: white; }}
        .btn-danger:hover {{ background: #b91c1c; }}

        .table-container {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: auto;
            max-height: 70vh;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        thead {{
            position: sticky;
            top: 0;
            z-index: 10;
            background: #f1f5f9;
        }}
        th {{
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border);
            cursor: pointer;
            white-space: nowrap;
        }}
        th:hover {{ background: #e2e8f0; }}
        td {{
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        tr:hover td {{ background: #f8fafc; }}
        tr.outlier td {{ background: #fef2f2; }}
        tr.outlier:hover td {{ background: #fee2e2; }}
        tr.discarded td {{ opacity: 0.5; text-decoration: line-through; }}
        tr.discarded:hover td {{ background: #fef2f2; }}

        .thumb {{
            width: 80px;
            height: 60px;
            object-fit: cover;
            border-radius: 4px;
            background: var(--border);
            cursor: pointer;
        }}
        .thumb:hover {{ opacity: 0.8; }}
        .thumb-placeholder {{
            width: 80px;
            height: 60px;
            background: #fee2e2;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            color: var(--danger);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        .badge-ok {{ background: #dcfce7; color: #166534; }}
        .badge-outlier {{ background: #fee2e2; color: #991b1b; }}
        .badge-empty {{ background: #ffedd5; color: #9a3412; }}
        .badge-keep {{ background: #dcfce7; color: #166534; }}
        .badge-discard {{ background: #fee2e2; color: #991b1b; }}
        .badge-review {{ background: #ffedd5; color: #9a3412; }}
        .badge-pending {{ background: #f1f5f9; color: var(--text-muted); }}

        code {{ font-family: 'SF Mono', Monaco, 'Courier New', monospace; font-size: 0.8rem; }}

        .review-select {{
            padding: 4px 8px;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 0.75rem;
            background: white;
            cursor: pointer;
        }}
        .review-select.keep {{ border-color: var(--success); background: #f0fdf4; }}
        .review-select.discard {{ border-color: var(--danger); background: #fef2f2; }}
        .review-select.review {{ border-color: var(--warning); background: #fffbeb; }}

        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.85);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}
        .modal.active {{ display: flex; }}
        .modal img {{
            max-width: 90%;
            max-height: 90%;
            border-radius: 8px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 20px;
            color: white;
            font-size: 2rem;
            cursor: pointer;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .modal-close:hover {{ background: rgba(255,255,255,0.3); }}

        .pagination {{
            display: flex;
            gap: 4px;
            margin-top: 12px;
            align-items: center;
            justify-content: center;
        }}
        .pagination button {{
            padding: 6px 12px;
            border: 1px solid var(--border);
            background: white;
            border-radius: 4px;
            cursor: pointer;
        }}
        .pagination button.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
        .pagination button:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Wound Dataset Review</h1>
            <div class="subtitle">Source: {csv_filename}</div>
        </header>

        <div class="stats-bar" id="statsBar"></div>

        <div class="filters-panel">
            <div class="filters-row">
                <input type="text" id="searchInput" placeholder="Search filename, source...">
                <select id="sourceFilter">
                    <option value="">All Sources</option>
                </select>
                <select id="splitFilter">
                    <option value="">All Splits</option>
                    <option value="train">Train</option>
                    <option value="test">Test</option>
                </select>
                <label><input type="checkbox" id="outliersOnly"> Outliers only</label>
                <label><input type="checkbox" id="pendingOnly"> Pending review</label>
            </div>
            <div class="filters-row" style="margin-top: 10px;">
                <div class="range-filter">
                    <label>Wound %:</label>
                    <div class="range-values">
                        <input type="number" id="woundMin" value="{wound_min_int}" min="0" max="100" step="0.1">
                        <span>-</span>
                        <input type="number" id="woundMax" value="{wound_max_int}" min="0" max="100" step="0.1">
                    </div>
                    <div class="dual-range" id="woundRangeContainer">
                        <input type="range" id="woundRangeMin" min="{wound_min_int}" max="{wound_max_int}" value="{wound_min_int}" step="0.1">
                        <input type="range" id="woundRangeMax" min="{wound_min_int}" max="{wound_max_int}" value="{wound_max_int}" step="0.1">
                    </div>
                </div>
                <div class="range-filter">
                    <label>Brightness:</label>
                    <div class="range-values">
                        <input type="number" id="brightMin" value="{bright_min_int}" min="0" max="255">
                        <span>-</span>
                        <input type="number" id="brightMax" value="{bright_max_int}" min="0" max="255">
                    </div>
                    <div class="dual-range" id="brightRangeContainer">
                        <input type="range" id="brightRangeMin" min="{bright_min_int}" max="{bright_max_int}" value="{bright_min_int}">
                        <input type="range" id="brightRangeMax" min="{bright_min_int}" max="{bright_max_int}" value="{bright_max_int}">
                    </div>
                </div>
            </div>
        </div>

        <div class="actions-bar">
            <button class="btn btn-primary" id="exportBtn">Export CSV reviewed</button>
            <button class="btn btn-secondary" id="resetBtn">Reset Reviews</button>
        </div>

        <div class="table-container">
            <table id="dataTable">
                <thead>
                    <tr>
                        <th>Original</th>
                        <th>Mask</th>
                        <th data-sort="filename">Filename</th>
                        <th data-sort="source">Source</th>
                        <th data-sort="split">Split</th>
                        <th data-sort="wound_percentage">Wound %</th>
                        <th data-sort="brightness_mean">Bright.</th>
                        <th data-sort="contrast_rms">Contrast</th>
                        <th data-sort="mask_edge_density">Edge Dens.</th>
                        <th data-sort="is_outlier">Status</th>
                        <th>Review</th>
                    </tr>
                </thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>

        <div class="pagination" id="pagination"></div>

        <footer>
            Wound Segmentation Dataset Review | SOLID Architecture
        </footer>
    </div>

    <div class="modal" id="imageModal">
        <span class="modal-close" id="modalClose">&times;</span>
        <img id="modalImage" src="" alt="">
    </div>

    <script>
        const DATA = {data_json};
        const STORAGE_KEY = 'wound_review_status_v2';
        const PAGE_SIZE = 50;
        const WOUND_MIN = {wound_min_int};
        const WOUND_MAX = {wound_max_int};
        const BRIGHT_MIN = {bright_min_int};
        const BRIGHT_MAX = {bright_max_int};

        let currentPage = 1;
        let sortColumn = 'filename';
        let sortAsc = true;
        let filters = {{
            search: '',
            source: '',
            split: '',
            outliersOnly: false,
            pendingOnly: false,
            woundMin: WOUND_MIN,
            woundMax: WOUND_MAX,
            brightMin: BRIGHT_MIN,
            brightMax: BRIGHT_MAX
        }};

        function getReviewStatus() {{
            try {{
                return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{}};
            }} catch {{ return {{}}; }}
        }}

        function saveReviewStatus(status) {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(status));
        }}

        function getStatus(idx) {{
            const status = getReviewStatus();
            return status[idx] || DATA[idx]?.review_status || '';
        }}

        function setStatus(idx, value) {{
            const status = getReviewStatus();
            if (value) status[idx] = value;
            else delete status[idx];
            saveReviewStatus(status);
            updateStats();
        }}

        function extractSource(filename) {{
            if (!filename) return 'unknown';
            const parts = String(filename).split('_');
            return parts[0].toLowerCase() || 'unknown';
        }}

        function isOutlier(row) {{
            return row.is_outlier === true || row.is_outlier === 1 || row.is_outlier === 'true';
        }}

        function isEmptyMask(row) {{
            return row.is_empty === true || row.is_empty === 1 || row.is_empty === 'true';
        }}

        function getUniqueSources() {{
            const sources = new Set();
            DATA.forEach(r => {{ sources.add(r.source || extractSource(r.filename)); }});
            return Array.from(sources).sort();
        }}

        function matchesFilters(row) {{
            const source = row.source || extractSource(row.filename);
            if (filters.source && source !== filters.source) return false;
            if (filters.split && row.split !== filters.split) return false;
            if (filters.outliersOnly && !isOutlier(row) && !isEmptyMask(row)) return false;
            if (filters.pendingOnly && getStatus(row._idx)) return false;

            const wound = parseFloat(row.wound_percentage) || 0;
            if (wound < filters.woundMin || wound > filters.woundMax) return false;

            const bright = parseFloat(row.brightness_mean) || 0;
            if (bright < filters.brightMin || bright > filters.brightMax) return false;

            if (filters.search) {{
                const s = filters.search.toLowerCase();
                const match = (String(row.filename || '').toLowerCase().includes(s) ||
                             source.toLowerCase().includes(s));
                if (!match) return false;
            }}
            return true;
        }}

        function getFilteredData() {{
            let filtered = DATA.filter(matchesFilters);
            filtered.sort((a, b) => {{
                let va = a[sortColumn] ?? '';
                let vb = b[sortColumn] ?? '';
                if (typeof va === 'string') va = va.toLowerCase();
                if (typeof vb === 'string') vb = vb.toLowerCase();
                if (va < vb) return sortAsc ? -1 : 1;
                if (va > vb) return sortAsc ? 1 : -1;
                return 0;
            }});
            return filtered;
        }}

        function updateStats() {{
            const total = DATA.length;
            const filtered = getFilteredData();
            const reviewStatus = getReviewStatus();
            let pending = 0;
            let kept = 0;
            let discarded = 0;
            let toReview = 0;

            DATA.forEach((r, i) => {{
                const status = reviewStatus[i] || r.review_status || '';
                if (!status) pending++;
                else if (status === 'keep') kept++;
                else if (status === 'discard') discarded++;
                else if (status === 'review') toReview++;
            }});

            const outliers = DATA.filter(r => isOutlier(r) || isEmptyMask(r)).length;

            document.getElementById('statsBar').innerHTML = `
                <div class="stat-pill"><span class="value">${{total}}</span> Total</div>
                <div class="stat-pill warning"><span class="value">${{outliers}}</span> Outliers</div>
                <div class="stat-pill"><span class="value">${{pending}}</span> Pending</div>
                <div class="stat-pill success"><span class="value">${{kept}}</span> Keep</div>
                <div class="stat-pill danger"><span class="value">${{discarded}}</span> Discard</div>
                <div class="stat-pill warning"><span class="value">${{toReview}}</span> Review</div>
                <div class="stat-pill"><span class="value">${{filtered.length}}</span> of ${{total}} showing</div>
            `;
        }}

        function renderPage() {{
            const filtered = getFilteredData();
            const start = (currentPage - 1) * PAGE_SIZE;
            const page = filtered.slice(start, start + PAGE_SIZE);

            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = page.map(row => {{
                const idx = row._idx;
                const source = row.source || extractSource(row.filename);
                const status = getStatus(idx);
                const outlier = isOutlier(row);
                const empty = isEmptyMask(row);

                let statusBadge = '<span class="badge badge-ok">OK</span>';
                let reasonText = '';
                if (outlier) {{
                    statusBadge = '<span class="badge badge-outlier">OUTLIER</span>';
                    if (row.outlier_reason) reasonText = ' - ' + row.outlier_reason;
                }}
                if (empty) {{
                    statusBadge = '<span class="badge badge-empty">EMPTY</span>';
                    reasonText = row.outlier_reason ? ' - ' + row.outlier_reason : ' - empty mask';
                }}

                let reviewClass = '';
                if (status === 'keep') reviewClass = 'keep';
                if (status === 'discard') reviewClass = 'discard';
                if (status === 'review') reviewClass = 'review';

                let rowClass = '';
                if (outlier || empty) rowClass = 'outlier';
                if (status === 'discard') rowClass = 'discard';

                const imgSrc = row.image_path || '';
                const maskSrc = row.mask_path || '';

                const edgeDensity = row.mask_edge_density ?? row.edge_density ?? 0;

                return `
                <tr class="${{rowClass}}">
                    <td>${{imgSrc ? `<img class="thumb" src="${{imgSrc}}" onclick="openModal('${{imgSrc.replace(/'/g, "\\\\'")}}')" alt="orig">` : '<div class="thumb-placeholder">N/A</div>'}}</td>
                    <td>${{maskSrc ? `<img class="thumb" src="${{maskSrc}}" onclick="openModal('${{maskSrc.replace(/'/g, "\\\\'")}}')" alt="mask">` : '<div class="thumb-placeholder">N/A</div>'}}</td>
                    <td><code>${{row.filename || ''}}</code></td>
                    <td>${{source}}</td>
                    <td>${{row.split || ''}}</td>
                    <td>${{row.wound_percentage != null ? row.wound_percentage.toFixed(2) + '%' : '-'}}</td>
                    <td>${{row.brightness_mean != null ? row.brightness_mean.toFixed(1) : '-'}}</td>
                    <td>${{row.contrast_rms != null ? row.contrast_rms.toFixed(2) : '-'}}</td>
                    <td>${{edgeDensity.toFixed(4)}}</td>
                    <td>${{statusBadge}}${{reasonText ? `<br><small style="color:var(--text-muted)">${{reasonText}}</small>` : ''}}</td>
                    <td>
                        <select class="review-select ${{reviewClass}}" onchange="setStatus(${{idx}}, this.value)">
                            <option value="">-</option>
                            <option value="keep" ${{status === 'keep' ? 'selected' : ''}}>Keep</option>
                            <option value="discard" ${{status === 'discard' ? 'selected' : ''}}>Discard</option>
                            <option value="review" ${{status === 'review' ? 'selected' : ''}}>Review</option>
                        </select>
                    </td>
                </tr>
                `;
            }}).join('');

            renderPagination(filtered.length);
            updateStats();
        }}

        function renderPagination(total) {{
            const pages = Math.ceil(total / PAGE_SIZE);
            const pagination = document.getElementById('pagination');
            if (pages <= 1) {{ pagination.innerHTML = ''; return; }}

            let html = `<button ${{currentPage === 1 ? 'disabled' : ''}} onclick="changePage(${{currentPage - 1}})">&laquo;</button>`;
            for (let i = 1; i <= pages; i++) {{
                if (i === 1 || i === pages || (i >= currentPage - 2 && i <= currentPage + 2)) {{
                    html += `<button class="${{i === currentPage ? 'active' : ''}}" onclick="changePage(${{i}})">${{i}}</button>`;
                }} else if (i === currentPage - 3 || i === currentPage + 3) {{
                    html += '<span style="padding:0 4px">...</span>';
                }}
            }}
            html += `<button ${{currentPage === pages ? 'disabled' : ''}} onclick="changePage(${{currentPage + 1}})">&raquo;</button>`;
            pagination.innerHTML = html;
        }}

        function changePage(p) {{
            currentPage = p;
            renderPage();
        }}

        function openModal(src) {{
            if (!src) return;
            document.getElementById('modalImage').src = src;
            document.getElementById('imageModal').classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('imageModal').classList.remove('active');
            document.getElementById('modalImage').src = '';
        }}

        function exportCSV() {{
            const reviewStatus = getReviewStatus();
            const headers = Object.keys(DATA[0]).filter(k => k !== '_idx');
            const rows = DATA.map((row, i) => {{
                const r = {{...row}};
                r.review_status = reviewStatus[i] || row.review_status || '';
                return r;
            }});

            let csv = headers.join(',') + '\\n';
            rows.forEach(r => {{
                csv += headers.map(h => {{
                    let v = r[h];
                    if (v === null || v === undefined) v = '';
                    if (typeof v === 'string' && (v.includes(',') || v.includes('"') || v.includes('\\n'))) {{
                        v = '"' + v.replace(/"/g, '""') + '"';
                    }}
                    return v;
                }}).join(',') + '\\n';
            }});

            const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dataset_reviewed.csv';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function resetReviews() {{
            if (confirm('Reset all review statuses? This will clear all localStorage data.')) {{
                localStorage.removeItem(STORAGE_KEY);
                document.querySelectorAll('.review-select').forEach(s => s.value = '');
                updateStats();
            }}
        }}

        function syncDualRange(minInput, maxInput, minVal, maxVal) {{
            const min = parseFloat(minInput.value);
            const max = parseFloat(maxInput.value);
            if (min > max) {{
                if (minInput === minInput) minInput.value = max;
                else maxInput.value = min;
            }}
            filters[minVal] = parseFloat(minInput.value);
            filters[maxVal] = parseFloat(maxInput.value);
            currentPage = 1;
            renderPage();
        }}

        function setupDualRange(minSliderId, maxSliderId, minInputId, maxInputId, minValKey, maxValKey) {{
            const minSlider = document.getElementById(minSliderId);
            const maxSlider = document.getElementById(maxSliderId);
            const minInput = document.getElementById(minInputId);
            const maxInput = document.getElementById(maxInputId);

            function handleMinChange() {{
                const val = parseFloat(minInput.value);
                if (val > parseFloat(maxInput.value)) {{
                    minInput.value = maxInput.value;
                }}
                filters[minValKey] = parseFloat(minInput.value);
                if (parseFloat(minSlider.value) > parseFloat(maxSlider.value)) {{
                    minSlider.value = maxSlider.value;
                }}
                currentPage = 1;
                renderPage();
            }}

            function handleMaxChange() {{
                const val = parseFloat(maxInput.value);
                if (val < parseFloat(minInput.value)) {{
                    maxInput.value = minInput.value;
                }}
                filters[maxValKey] = parseFloat(maxInput.value);
                if (parseFloat(maxSlider.value) < parseFloat(minSlider.value)) {{
                    maxSlider.value = minSlider.value;
                }}
                currentPage = 1;
                renderPage();
            }}

            function handleMinSlider() {{
                const minVal = parseFloat(minSlider.value);
                const maxVal = parseFloat(maxSlider.value);
                if (minVal > maxVal) {{
                    if (minSlider === minSlider) minSlider.value = maxVal;
                    else maxSlider.value = minVal;
                }}
                minInput.value = minSlider.value;
                filters[minValKey] = minVal;
                currentPage = 1;
                renderPage();
            }}

            function handleMaxSlider() {{
                const minVal = parseFloat(minSlider.value);
                const maxVal = parseFloat(maxSlider.value);
                if (maxVal < minVal) {{
                    if (maxSlider === maxSlider) maxSlider.value = minVal;
                    else minSlider.value = maxVal;
                }}
                maxInput.value = maxSlider.value;
                filters[maxValKey] = maxVal;
                currentPage = 1;
                renderPage();
            }}

            minSlider.addEventListener('input', handleMinSlider);
            maxSlider.addEventListener('input', handleMaxSlider);
            minInput.addEventListener('change', handleMinChange);
            maxInput.addEventListener('change', handleMaxChange);
        }}

        // Init
        document.addEventListener('DOMContentLoaded', () => {{
            // Populate source filter
            const sourceSelect = document.getElementById('sourceFilter');
            getUniqueSources().forEach(s => {{
                sourceSelect.innerHTML += `<option value="${{s}}">${{s}}</option>`;
            }});

            // Setup dual range sliders
            setupDualRange('woundRangeMin', 'woundRangeMax', 'woundMin', 'woundMax', 'woundMin', 'woundMax');
            setupDualRange('brightRangeMin', 'brightRangeMax', 'brightMin', 'brightMax', 'brightMin', 'brightMax');

            // Sort handlers
            document.querySelectorAll('th[data-sort]').forEach(th => {{
                th.onclick = () => {{
                    const col = th.dataset.sort;
                    if (sortColumn === col) sortAsc = !sortAsc;
                    else {{ sortColumn = col; sortAsc = true; }}
                    renderPage();
                }}
            }});

            // Filter handlers
            document.getElementById('searchInput').oninput = e => {{ filters.search = e.target.value; currentPage = 1; renderPage(); }};
            document.getElementById('sourceFilter').onchange = e => {{ filters.source = e.target.value; currentPage = 1; renderPage(); }};
            document.getElementById('splitFilter').onchange = e => {{ filters.split = e.target.value; currentPage = 1; renderPage(); }};
            document.getElementById('outliersOnly').onchange = e => {{ filters.outliersOnly = e.target.checked; currentPage = 1; renderPage(); }};
            document.getElementById('pendingOnly').onchange = e => {{ filters.pendingOnly = e.target.checked; currentPage = 1; renderPage(); }};

            document.getElementById('exportBtn').onclick = exportCSV;
            document.getElementById('resetBtn').onclick = resetReviews;
            document.getElementById('modalClose').onclick = closeModal;
            document.getElementById('imageModal').onclick = e => {{ if (e.target.id === 'imageModal') closeModal(); }};

            // Keyboard to close modal
            document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

            renderPage();
        }});
    </script>
</body>
</html>"""