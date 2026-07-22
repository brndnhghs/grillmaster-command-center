(function() {
  const menu = document.getElementById('tl-ctx-menu');
  const menuMode = document.getElementById('tl-ctx-mode');
  const menuSplit = document.getElementById('tl-ctx-split');
  const menuDelete = document.getElementById('tl-ctx-delete');
  let ctxClipId = null;

  function showCtxMenu(x, y, clipId) {
    ctxClipId = clipId;
    const clip = tlClips.find(c => c.id === clipId);
    if (clip) {
      menuMode.textContent = clip.looped ? '✂ Switch to Trim' : '↺ Switch to Loop';
      const tf = parseInt(document.getElementById('tl-frame')?.value) || 0;
      const playheadInside = tf > clip.startFrame && tf <= clip.endFrame;
      menuSplit.style.display = playheadInside ? '' : 'none';
    }
    menu.style.display = 'block';
    const mw = menu.offsetWidth, mh = menu.offsetHeight;
    menu.style.left = Math.min(x, window.innerWidth - mw - 4) + 'px';
    menu.style.top  = Math.min(y, window.innerHeight - mh - 4) + 'px';
  }
  window.tlShowClipCtxMenu = showCtxMenu;

  function hideCtxMenu() { menu.style.display = 'none'; ctxClipId = null; }

  document.addEventListener('click', hideCtxMenu);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideCtxMenu(); });

  menuMode.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!ctxClipId) return;
    const clip = tlClips.find(c => c.id === ctxClipId);
    if (!clip) return;
    clip.looped = !clip.looped;
    if (!clip.looped) {
      // Switching to trim: snap right edge back within source bounds
      const us = Math.max(1, (clip.srcLength || 1) - (clip.trimIn || 0));
      clip.endFrame = Math.min(clip.endFrame, clip.startFrame + us - 1);
      clip._origEnd = clip.endFrame;
    }
    renderTimelineRuler();
    tlSaveClips();
    hideCtxMenu();
  });

  menuSplit.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!ctxClipId) return;
    tlSplitClipAtPlayhead(ctxClipId);
    hideCtxMenu();
  });

  menuDelete.addEventListener('click', () => {
    if (ctxClipId) tlDeleteClip(ctxClipId);
    hideCtxMenu();
  });

  // Delegate contextmenu on clip bars (right-click desktop; long-press fires contextmenu on mobile)
  document.addEventListener('contextmenu', (e) => {
    const bar = e.target.closest('.tl-clip-bar');
    if (!bar) return;
    e.preventDefault();
    tlSelectClip(bar.dataset.clipId);
    showCtxMenu(e.clientX, e.clientY, bar.dataset.clipId);
  });
})();
