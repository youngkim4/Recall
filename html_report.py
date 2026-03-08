#!/usr/bin/env python3
"""
Module: html_report

Purpose:
- Generate HTML reports with charts for conversation analysis
"""
import html as html_lib
import os
import json
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import markdown


def write_html_report(
    out_dir: str,
    contact: str,
    stats,  # ConversationStats
    monthly: pd.DataFrame,
    summary_text: str,
    sentiment: List = None,
    events_df: pd.DataFrame = None
) -> str:
    """Generate an HTML report with charts."""
    from analysis import sanitize_filename

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"analysis_{sanitize_filename(contact)}.html")

    # Prepare chart data
    months_labels = [str(row["month"].date()) for _, row in monthly.iterrows()]
    sent_data = [int(row["sent"]) for _, row in monthly.iterrows()]
    received_data = [int(row["received"]) for _, row in monthly.iterrows()]

    # Events section
    events_html = ""
    if events_df is not None and not events_df.empty:
        events_cards = []
        for _, ev in events_df.iterrows():
            quote_block = ""
            if ev.get('quote'):
                quote_block = (
                    f'<blockquote class="event-quote">'
                    f'"{html_lib.escape(str(ev["quote"]))}"'
                    f'</blockquote>'
                )
            events_cards.append(
                f'<div class="event">'
                f'  <div class="event-header">'
                f'    <span class="event-date">{html_lib.escape(str(ev["date"]))}</span>'
                f'    <span class="event-category">{html_lib.escape(str(ev["category"]))}</span>'
                f'  </div>'
                f'  <div class="event-title">{html_lib.escape(str(ev["title"]))}</div>'
                f'  <div class="event-detail">{html_lib.escape(str(ev["detail"]))}</div>'
                f'  {quote_block}'
                f'</div>'
            )
        events_html = (
            '<section class="section">'
            '  <h2>Key Events</h2>'
            '  <div class="events">' + "\n".join(events_cards) + '</div>'
            '</section>'
        )

    # Sentiment section
    sentiment_html = ""
    if sentiment:
        bars = []
        for pt in sentiment:
            if pt.score > 0.2:
                color = 'var(--tone-high)'
            elif pt.score > -0.2:
                color = 'var(--tone-mid)'
            else:
                color = 'var(--tone-low)'
            width = (pt.score + 1) * 50
            bars.append(
                f'<div class="sentiment-bar">'
                f'  <span class="sentiment-period">{pt.month}</span>'
                f'  <div class="sentiment-fill">'
                f'    <div class="sentiment-fill-inner" style="width: {width}%; background: {color};"></div>'
                f'  </div>'
                f'  <span class="sentiment-summary">{pt.summary[:50]}...</span>'
                f'</div>'
            )
        sentiment_html = (
            '<section class="section">'
            '  <h2>Sentiment Over Time</h2>'
            '  <div class="card">' + "\n".join(bars) + '</div>'
            '</section>'
        )

    # Attachments section
    attachments_html = ""
    if stats.attachments.total > 0:
        attachments_html = (
            '<section class="section">'
            '  <h2>Attachments</h2>'
            '  <div class="stats-grid">'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.attachments.photos:,}</div>'
            '      <div class="stat-label">Photos</div>'
            '    </div>'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.attachments.videos:,}</div>'
            '      <div class="stat-label">Videos</div>'
            '    </div>'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.attachments.audio:,}</div>'
            '      <div class="stat-label">Audio</div>'
            '    </div>'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.attachments.gifs:,}</div>'
            '      <div class="stat-label">GIFs</div>'
            '    </div>'
            '  </div>'
            '</section>'
        )

    # Reactions section
    reactions_html = ""
    if stats.reactions.total > 0:
        reactions_html = (
            '<section class="section">'
            '  <h2>Reactions</h2>'
            '  <div class="stats-grid">'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.reactions.loves:,}</div>'
            '      <div class="stat-label">Loves</div>'
            '    </div>'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.reactions.likes:,}</div>'
            '      <div class="stat-label">Likes</div>'
            '    </div>'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.reactions.laughs:,}</div>'
            '      <div class="stat-label">Laughs</div>'
            '    </div>'
            '    <div class="stat-card">'
            f'      <div class="stat-value">{stats.reactions.emphasis:,}</div>'
            '      <div class="stat-label">Emphasis</div>'
            '    </div>'
            '  </div>'
            '</section>'
        )

    summary_rendered = markdown.markdown(summary_text, extensions=['extra'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Analysis — {html_lib.escape(str(contact))}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #0a0a0a;
            --bg-raised: #0a0a0a;
            --bg-surface: #141414;
            --border-subtle: rgba(255, 255, 255, 0.04);
            --text-primary: #e8e8e8;
            --text-secondary: #888888;
            --text-tertiary: #505050;
            --tone-high: rgba(255, 255, 255, 0.70);
            --tone-mid: rgba(255, 255, 255, 0.35);
            --tone-low: rgba(255, 255, 255, 0.15);
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 12px;
            --space-4: 16px;
            --space-5: 24px;
            --space-6: 32px;
            --space-7: 48px;
            --space-8: 64px;
            --space-9: 96px;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-base);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        .container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 0 var(--space-5);
        }}

        /* Header */
        header {{
            text-align: center;
            padding: var(--space-9) 0 var(--space-7);
            margin-bottom: var(--space-8);
        }}

        h1 {{
            font-size: 2.75rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            color: var(--text-primary);
            margin-bottom: var(--space-3);
        }}

        .subtitle {{
            color: var(--text-tertiary);
            font-size: 0.875rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }}

        /* Sections */
        .section {{
            margin-bottom: var(--space-9);
        }}

        h2 {{
            font-size: 0.6875rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-tertiary);
            margin-bottom: var(--space-5);
        }}

        /* Stats grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: var(--space-6);
        }}

        .stat-card {{
            padding: var(--space-4) 0;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }}

        .stat-label {{
            color: var(--text-tertiary);
            font-size: 0.6875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: var(--space-2);
        }}

        /* Card */
        .card {{
            padding: var(--space-5);
        }}

        /* Chart */
        .chart-container {{
            padding: var(--space-4) 0;
        }}

        /* Events */
        .events {{
            display: grid;
            gap: 0;
        }}

        .event {{
            padding: var(--space-5) 0;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .event:last-child {{
            border-bottom: none;
        }}

        .event-header {{
            display: flex;
            align-items: center;
            gap: var(--space-3);
            margin-bottom: var(--space-2);
        }}

        .event-date {{
            color: var(--text-tertiary);
            font-size: 0.75rem;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }}

        .event-category {{
            display: inline-block;
            color: var(--text-tertiary);
            font-size: 0.625rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .event-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: var(--space-1);
        }}

        .event-detail {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        .event-quote {{
            border-left: 1px solid rgba(255, 255, 255, 0.12);
            padding-left: var(--space-4);
            margin-top: var(--space-3);
            color: var(--text-secondary);
            font-style: italic;
            font-size: 0.875rem;
        }}

        /* Sentiment */
        .sentiment-bar {{
            display: flex;
            align-items: center;
            gap: var(--space-4);
            padding: var(--space-3) 0;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .sentiment-bar:last-child {{ border-bottom: none; }}

        .sentiment-period {{
            width: 80px;
            font-weight: 600;
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }}

        .sentiment-fill {{
            flex: 1;
            height: 14px;
            background: var(--bg-surface);
            border-radius: 7px;
            overflow: hidden;
        }}

        .sentiment-fill-inner {{
            height: 100%;
            border-radius: 7px;
            transition: width 0.3s ease;
        }}

        .sentiment-summary {{
            width: 180px;
            color: var(--text-tertiary);
            font-size: 0.75rem;
        }}

        /* Summary */
        .summary {{
            padding: var(--space-6) 0;
            line-height: 1.8;
        }}

        .summary > * {{
            max-width: 65ch;
        }}

        .summary h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: var(--space-7);
            margin-bottom: var(--space-3);
            letter-spacing: -0.02em;
        }}

        .summary h3 {{
            font-size: 1.0625rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: var(--space-5);
            margin-bottom: var(--space-2);
        }}

        .summary p {{
            color: var(--text-secondary);
            margin-bottom: var(--space-4);
        }}

        .summary ul, .summary ol {{
            margin: var(--space-3) 0;
            padding-left: var(--space-5);
            color: var(--text-secondary);
        }}

        .summary li {{
            margin-bottom: var(--space-2);
        }}

        .summary blockquote {{
            border-left: 1px solid rgba(255, 255, 255, 0.15);
            padding-left: var(--space-4);
            margin: var(--space-4) 0;
            color: var(--text-secondary);
            font-style: italic;
        }}

        .summary strong {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        .summary em {{
            color: var(--text-secondary);
        }}

        .summary hr {{
            border: none;
            border-top: 1px solid var(--border-subtle);
            margin: var(--space-7) 0;
        }}

        .summary code {{
            background: var(--bg-surface);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.875em;
        }}

        /* Footer */
        footer {{
            text-align: center;
            padding: var(--space-7) 0;
            color: var(--text-tertiary);
            font-size: 0.6875rem;
            font-weight: 500;
            letter-spacing: 0.02em;
            margin-top: var(--space-9);
        }}

        /* Responsive */
        @media (max-width: 640px) {{
            h1 {{ font-size: 2rem; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); gap: var(--space-4); }}
            .stat-value {{ font-size: 1.75rem; }}
            .sentiment-summary {{ display: none; }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{html_lib.escape(str(contact))}</h1>
            <p class="subtitle">Conversation analysis from {html_lib.escape(str(stats.first_timestamp))} to {html_lib.escape(str(stats.last_timestamp))}</p>
        </div>
    </header>

    <main class="container">
        <section class="section">
            <h2>Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.total_messages:,}</div>
                    <div class="stat-label">Total Messages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.sent_count:,}</div>
                    <div class="stat-label">Sent</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.received_count:,}</div>
                    <div class="stat-label">Received</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.active_days:,}</div>
                    <div class="stat-label">Active Days</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.avg_messages_per_day:.1f}</div>
                    <div class="stat-label">Avg / Day</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.longest_gap_days:.0f}</div>
                    <div class="stat-label">Longest Gap (days)</div>
                </div>
            </div>
        </section>

        {attachments_html}

        {reactions_html}

        <section class="section">
            <h2>Message Volume Over Time</h2>
            <div class="chart-container">
                <canvas id="volumeChart"></canvas>
            </div>
        </section>

        {sentiment_html}

        {events_html}

        <section class="section">
            <h2>Summary</h2>
            <div class="summary">{summary_rendered}</div>
        </section>
    </main>

    <footer>
        <div class="container">
            <p>Generated by iMessage Memories &middot; {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    </footer>

    <script>
        const ctx = document.getElementById('volumeChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(months_labels)},
                datasets: [
                    {{
                        label: 'Sent',
                        data: {json.dumps(sent_data)},
                        borderColor: 'rgba(255, 255, 255, 0.7)',
                        backgroundColor: 'rgba(255, 255, 255, 0.03)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointHoverBackgroundColor: '#e8e8e8',
                        pointHoverBorderColor: '#e8e8e8',
                        tension: 0.3,
                        fill: true,
                    }},
                    {{
                        label: 'Received',
                        data: {json.dumps(received_data)},
                        borderColor: 'rgba(255, 255, 255, 0.25)',
                        backgroundColor: 'rgba(255, 255, 255, 0.01)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointHoverBackgroundColor: '#888888',
                        pointHoverBorderColor: '#888888',
                        tension: 0.3,
                        fill: true,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                interaction: {{
                    intersect: false,
                    mode: 'index',
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#505050',
                            font: {{ family: "'Inter', sans-serif", size: 11, weight: '600' }},
                            boxWidth: 10,
                            boxHeight: 10,
                            borderRadius: 0,
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'line',
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: '#141414',
                        titleColor: '#e8e8e8',
                        bodyColor: '#888888',
                        borderColor: 'rgba(255, 255, 255, 0.06)',
                        borderWidth: 1,
                        titleFont: {{ family: "'Inter', sans-serif", size: 11, weight: '600' }},
                        bodyFont: {{ family: "'Inter', sans-serif", size: 11 }},
                        padding: 12,
                        cornerRadius: 4,
                        displayColors: false,
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.03)', drawBorder: false }},
                        ticks: {{
                            color: '#404040',
                            font: {{ family: "'Inter', sans-serif", size: 10, weight: '500' }},
                            maxRotation: 45,
                        }},
                        border: {{ display: false }},
                    }},
                    y: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.03)', drawBorder: false }},
                        ticks: {{
                            color: '#404040',
                            font: {{ family: "'Inter', sans-serif", size: 10, weight: '500' }},
                        }},
                        border: {{ display: false }},
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    return fname
