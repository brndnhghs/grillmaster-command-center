(function() {
  const ntPanel      = document.getElementById('nt-panel');
  const ntCloseBtn   = document.getElementById('nt-close-btn');
  const ntResults    = document.getElementById('nt-results');
  const ntProgress   = document.getElementById('nt-progress');
  const ntProgressText = document.getElementById('nt-progress-text');
  const ntSummary    = document.getElementById('nt-summary');
  const ntRunBtn     = document.getElementById('nt-run-btn');
  const ntBatchBtn   = document.getElementById('nt-batch-apply-btn');
  const ntExportBtn  = document.getElementById('nt-export-btn');
  const ntFooter     = document.getElementById('nt-footer');
  const testBtnDesk  = document.getElementById('graph-test-btn-desk');

  let ntReport = null;
  let ntRunning = false;
  let ntFixes = {};  // method_id → source_code from Node Doctor

  function ntOpen() {
    ntPanel.classList.add('nt-open');
    ntLoadLastReport();
  }
  function ntClose() { ntPanel.classList.remove('nt-open'); }
  ntCloseBtn.addEventListener('click', ntClose);
  testBtnDesk.addEventListener('click', ntOpen);

  async function ntLoadLastReport() {
    try {
      const r = await fetch('/api/node-tester/report');
      const d = await r.json();
      if (d.report) {
        ntReport = d.report;
        ntRenderReport(d.report);
      } else {
        ntResults.innerHTML = '<div style="color:var(--muted);font-size:12px;padding:12px 0">No test report yet. Click "Run All Tests" to start.</div>';
        ntSummary.textContent = '';
        ntFooter.style.display = 'flex';
      }
    } catch {
      ntResults.innerHTML = '<div style="color:var(--err);font-size:12px">Could not load report</div>';
    }
  }

  function ntRenderReport(report) {
    ntResults.innerHTML = '';
    ntSummary.textContent = `${report.passed}✓ ${report.failed}✗ (${report.duration_s}s)`;
    ntFooter.style.display = 'flex';

    if (!report.results || report.results.length === 0) {
      ntResults.innerHTML = '<div style="color:var(--muted);font-size:12px;padding:12px 0">No results.</div>';
      return;
    }

    // Group by method_id
    const byMethod = {};
    for (const r of report.results) {
      if (!byMethod[r.method_id]) byMethod[r.method_id] = { meta: r, tests: [] };
      byMethod[r.method_id].tests.push(r);
    }

    const hasFails = report.failed > 0;
    ntBatchBtn.style.display = hasFails ? '' : 'none';

    for (const [mid, group] of Object.entries(byMethod).sort()) {
      const anyFail = group.tests.some(t => !t.passed);
      const row = document.createElement('div');
      row.className = `nt-method-row ${anyFail ? 'nt-fail' : 'nt-pass'}`;
      row.innerHTML = `
        <span class="nt-status">${anyFail ? '✗' : '✓'}</span>
        <span class="nt-mid">${mid}</span>
        <span class="nt-name">${escHtml(group.meta.method_name)}</span>
        <span class="nt-detail">${group.tests.filter(t => !t.passed).length}/${group.tests.length} failed</span>
      `;

      // Detail expandable
      const detail = document.createElement('div');
      detail.className = 'nt-test-detail';
      for (const t of group.tests) {
        const p = document.createElement('div');
        p.innerHTML = `<span class="nt-param-set">${t.param_set}</span> — ${t.passed ? '✓' : '✗'} (${t.duration_ms}ms)`;
        detail.appendChild(p);
        if (!t.passed && t.error_trace) {
          const trace = document.createElement('div');
          trace.className = 'nt-error-text';
          trace.textContent = t.error_trace;
          detail.appendChild(trace);
        }
      }

      row.addEventListener('click', () => detail.classList.toggle('nt-open'));
      ntResults.appendChild(row);
      ntResults.appendChild(detail);
    }
  }

  ntRunBtn.addEventListener('click', async () => {
    if (ntRunning) return;
    ntRunning = true;
    ntRunBtn.disabled = true;
    ntRunBtn.textContent = '⏳ Running…';
    ntProgress.style.display = 'flex';
    ntProgressText.textContent = 'Starting…';
    ntResults.innerHTML = '';

    try {
      const resp = await fetch('/api/node-tester/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_edge_cases: true }),
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const d = JSON.parse(line.slice(6));
            if (d.done) {
              if (d.report) {
                ntReport = d.report;
                ntRenderReport(d.report);
                ntProgress.style.display = 'none';
              } else if (d.error) {
                ntResults.innerHTML = `<div style="color:var(--err);font-size:12px">⚠ ${d.error}</div>`;
                ntProgress.style.display = 'none';
              }
            }
          } catch { /* partial */ }
        }
      }
    } catch (err) {
      ntResults.innerHTML = `<div style="color:var(--err);font-size:12px">⚠ ${err.message}</div>`;
      ntProgress.style.display = 'none';
    }

    ntRunning = false;
    ntRunBtn.disabled = false;
    ntRunBtn.textContent = '▶ Run All Tests';
  });

  ntExportBtn.addEventListener('click', () => {
    if (!ntReport) return;
    const blob = new Blob([JSON.stringify(ntReport, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `node-test-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // Batch apply: collect Node Doctor fixes for all failing methods
  ntBatchBtn.addEventListener('click', async () => {
    if (!ntReport || !ntReport.failed) return;
    // For each failing method, try to get a fix from Node Doctor
    const failingIds = new Set();
    for (const r of ntReport.results) {
      if (!r.passed) failingIds.add(r.method_id);
    }
    if (failingIds.size === 0) return;

    // We need source code for each failing method — fetch from node-doctor source endpoint
    const fixes = [];
    for (const mid of failingIds) {
      try {
        const r = await fetch(`/api/node-doctor/source/${mid}`);
        const d = await r.json();
        if (d.source) {
          fixes.push({ method_id: mid, source: d.source });
        }
      } catch {}
    }

    if (fixes.length === 0) {
      gShowToast('No source files found for failing methods', true);
      return;
    }

    ntBatchBtn.disabled = true;
    ntBatchBtn.textContent = '⏳ Applying…';

    try {
      const r = await fetch('/api/node-tester/batch-apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fixes }),
      });
      const d = await r.json();
      gShowToast(`Applied ${d.applied} fixes, ${d.failed.length} failed`);
      if (d.applied > 0) {
        // Re-run tests after applying
        setTimeout(() => ntRunBtn.click(), 1000);
      }
    } catch (err) {
      gShowToast('Batch apply failed: ' + err.message, true);
    }

    ntBatchBtn.disabled = false;
    ntBatchBtn.textContent = '⚕ Batch Apply Fixes';
  });

  // Listen for test-progress SSE events
  const ntEs = new EventSource('/api/events');
  ntEs.addEventListener('test-progress', e => {
    try {
      const d = JSON.parse(e.data);
      ntProgressText.textContent = `${d.method_name} (${d.method_id}) — ${d.param_set}: ${d.status}`;
    } catch {}
  });
})();
