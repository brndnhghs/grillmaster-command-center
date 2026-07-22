(function () {
  const btn = document.getElementById('server-restart-btn');
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.classList.add('restarting');
    btn.title = 'Restarting…';

    try {
      await fetch('/admin/restart', { method: 'POST' });
    } catch (_) {
      // Server may close the connection before responding — that's fine
    }

    // Poll /health every 500 ms; reload once it responds ok (max 30 s)
    let attempts = 0;
    const poll = async () => {
      if (++attempts > 60) {
        btn.disabled = false;
        btn.classList.remove('restarting');
        btn.title = 'Server did not come back — check the terminal';
        return;
      }
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 2000);
        const r = await fetch('/health', { cache: 'no-store', signal: ctrl.signal });
        clearTimeout(t);
        if (r.ok) { window.location.reload(); return; }
      } catch (_) {}
      setTimeout(poll, 500);
    };
    // Give the old process ~1.2 s to die before we start polling
    setTimeout(poll, 1200);
  });
})();
