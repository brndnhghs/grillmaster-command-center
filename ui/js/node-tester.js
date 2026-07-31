(function() {
  // Tester lives inline inside the Node Doctor tab of the source-code modal.
  // The toolbar 🧪 button opens the modal on the Doctor tab.
  // Results only appear after explicitly clicking "Run All Tests" or "Test Current Node"
  // — no stale auto-population.

  const ntSection      = document.getElementById('nt-section');
  const ntHeader       = document.getElementById('nt-section-header');
  const ntToggle       = document.getElementById('nt-toggle');
  const ntSummary      = document.getElementById('nt-summary');
  const ntContextLabel = document.getElementById('nt-context-label');
  const ntResults      = document.getElementById('nt-results');
  const ntProgress     = document.getElementById('nt-progress');
  const ntProgressText = document.getElementById('nt-progress-text');
  const ntRunBtn       = document.getElementById('nt-run-btn');
  const ntRunCurrentBtn = document.getElementById('nt-run-current-btn');
  const ntExportBtn    = document.getElementById('nt-export-btn');
  let ntReport  = null;
  let ntRunning = false;

  // ── Collapse / expand ──
  ntHeader.addEventListener('click', () => {
    ntSection.classList.toggle('nt-collapsed');
    ntToggle.textContent = ntSection.classList.contains('nt-collapsed') ? '▶' : '▼';
  });

  // ── Open: called from toolbar button and from Node Doctor context ──
  function ntOpen(methodId) {
    ntSection.classList.remove('nt-collapsed');
    ntToggle.textContent = '▼';

    // Show/hide context label and per-node test button
    const node = gNodes.find(n => n.id === gSelectedNode);
    if (node && node.method_id) {
      const def = gNodeDefs[node.method_id];
      ntContextLabel.textContent = def ? `for ${def.name}` : `for ${node.method_id}`;
      ntRunCurrentBtn.style.display = '';
    } else if (methodId) {
      const def = gNodeDefs && gNodeDefs[methodId];
      ntContextLabel.textContent = def ? `for ${def.name}` : `for ${methodId}`;
      ntRunCurrentBtn.style.display = '';
    } else {
      ntContextLabel.textContent = '';
      ntRunCurrentBtn.style.display = 'none';
    }

    // Blank state — no stale auto-population
    ntResults.innerHTML = '<div style="color:var(--muted);font-size:12px;padding:12px 0">Click "Run All Tests" or "Test Current Node" to run.</div>';
    ntSummary.textContent = '';
  }
  window.ntOpen = ntOpen;

  // ── Render report ──
  function ntRenderReport(report) {
    ntResults.innerHTML = '';
    ntSummary.textContent = `${report.passed}✓ ${report.failed}✗ (${report.duration_s}s)`;

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

      // Expandable detail
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

  // ── Run tests (all or single method) ──
  async function ntRunTests(methodIds) {
    if (ntRunning) return;
    ntRunning = true;
    ntRunBtn.disabled = true;
    ntRunBtn.textContent = '⏳ Running…';
    if (ntRunCurrentBtn) { ntRunCurrentBtn.disabled = true; ntRunCurrentBtn.textContent = '⏳ Testing…'; }
    ntProgress.style.display = 'flex';
    ntProgressText.textContent = 'Starting…';
    ntResults.innerHTML = '';

    try {
      const resp = await fetch('/api/node-tester/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          include_edge_cases: true,
          method_ids: methodIds || null,
        }),
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
    if (ntRunCurrentBtn) { ntRunCurrentBtn.disabled = false; ntRunCurrentBtn.textContent = '▶ Test Current Node'; }
  }

  ntRunBtn.addEventListener('click', () => {
    if (ntRunning) return;
    if (!confirm('Run all methods through the test suite? This can take a few minutes for 180+ nodes.')) return;
    ntRunTests(null);
  });

  ntRunCurrentBtn.addEventListener('click', () => {
    const node = gNodes.find(n => n.id === gSelectedNode);
    if (node && node.method_id) {
      ntRunTests([node.method_id]);
    }
  });

  // ── Export ──
  ntExportBtn.addEventListener('click', () => {
    if (!ntReport) return;
    const blob = new Blob([JSON.stringify(ntReport, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `node-test-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // ── Live progress via SSE ──
  const ntEs = new EventSource('/api/events');
  ntEs.addEventListener('test-progress', e => {
    try {
      const d = JSON.parse(e.data);
      ntProgressText.textContent = `${d.method_name} (${d.method_id}) — ${d.param_set}: ${d.status}`;
    } catch {}
  });

  // ── When the modal opens on Doctor tab, refresh context (no stale load) ──
  const modal = document.getElementById('node-source-modal');
  if (modal) {
    const observer = new MutationObserver(() => {
      if (modal.classList.contains('open')) {
        const doctorTab = document.querySelector('.nsm-tab-doctor');
        if (doctorTab && doctorTab.classList.contains('active')) {
          ntOpen();
        }
      }
    });
    observer.observe(modal, { attributes: true, attributeFilter: ['class'] });
  }
})();
