(function (root) {
  'use strict';
  const actions = new Set(['fight', 'wrestle', 'fall', 'normal', 'hug', 'greet', 'sit_lie', 'sport_play', 'uncertain']);
  function validateIntervals(intervals, duration) {
    if (!Array.isArray(intervals)) throw Error('Thiếu danh sách đoạn.');
    let previous = 0;
    for (const interval of intervals) {
      if (!interval || !actions.has(interval.action) || !Number.isFinite(interval.start_seconds) || !Number.isFinite(interval.end_seconds) || interval.start_seconds < previous || interval.end_seconds <= interval.start_seconds || interval.end_seconds > duration + 1e-6) throw Error('Mốc thời gian hoặc nhãn không hợp lệ/chồng lấn.');
      previous = interval.end_seconds;
    }
  }
  function restore(template, saved, rows) {
    if (!saved || saved.profile !== template.profile || saved.schema_version !== template.schema_version || saved.manifest_sha256 !== template.manifest_sha256 || !Array.isArray(saved.sources)) throw Error('File nhãn không thuộc bộ dữ liệu này.');
    const document = JSON.parse(JSON.stringify(template)), entries = new Map(), originals = new Map(rows.map(row => [row.video, row]));
    for (const entry of saved.sources) {
      const source = originals.get(entry.video);
      if (!source || entries.has(entry.video) || entry.source_sha256 !== source.source_sha256 || entry.split !== source.split) throw Error('Nguồn/split đã thay đổi hoặc trùng lặp.');
      validateIntervals(entry.intervals, source.duration);
      if (!['pending', 'reviewed'].includes(entry.review_status) || (entry.review_status === 'reviewed' && (!entry.intervals.length || typeof entry.reviewer !== 'string' || !entry.reviewer.trim()))) throw Error('Thông tin người duyệt chưa hợp lệ.');
      entries.set(entry.video, entry);
    }
    document.sources = document.sources.map(entry => entries.has(entry.video) ? JSON.parse(JSON.stringify(entries.get(entry.video))) : entry);
    return document;
  }
  const api = {validateIntervals, restore};
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;
  const el = id => document.getElementById(id);
  let pack, labels, selected, key;
  function entry() { return labels.sources.find(item => item.video === selected.video); }
  function persist() {
    try { localStorage.setItem(key, JSON.stringify(labels)); } catch { el('message').textContent = 'Không lưu tự động được. Hãy tải file nhãn trước khi đóng trang.'; }
    const count = labels.sources.filter(item => item.review_status === 'reviewed').length;
    el('summary').textContent = `${count}/${labels.sources.length} nguồn đã duyệt · ${pack.missing_media.length} nguồn thiếu preview · Nhãn dương: đánh nhau/vật nhau/ngã nói chung. Chưa bắt đầu training.`;
  }
  function render() {
    const current = entry();
    el('intervals').replaceChildren();
    for (const [index, interval] of current.intervals.entries()) {
      const item = document.createElement('li');
      item.textContent = `${interval.start_seconds.toFixed(3)}–${interval.end_seconds.toFixed(3)} s · ${interval.action}${interval.evidence ? ' · ' + interval.evidence : ''} `;
      const remove = document.createElement('button'); remove.textContent = 'Xóa';
      remove.onclick = () => { current.intervals.splice(index, 1); current.review_status = 'pending'; persist(); render(); };
      item.append(remove); el('intervals').append(item);
    }
    el('reviewer').value = current.reviewer || '';
    const missing = pack.missing_media.includes(selected.video);
    el('message').textContent = `${current.review_status === 'reviewed' ? 'Đã duyệt' : 'Chờ duyệt'}${missing ? ' · Chưa có preview cho nguồn này' : ''}`;
    el('add').disabled = missing; el('reviewed').disabled = missing || !current.intervals.length;
  }
  function select() {
    selected = pack.manifest.sources[Number(el('source').value)];
    if (!/^media\/[a-f0-9]{24}\.mp4$/.test(selected.media)) throw Error('Đường dẫn preview không hợp lệ.');
    el('video').removeAttribute('src');
    if (!pack.missing_media.includes(selected.video)) el('video').src = selected.media;
    el('video').load(); el('start').value = 0; el('end').value = 0;
    el('identity').textContent = `${selected.dataset} · ${selected.split} · ${selected.duration.toFixed(3)} giây · ${selected.from_parent_v1 ? 'Nguồn v1, giữ split' : 'Nguồn bổ sung'}`;
    render();
  }
  function guard(action) { try { action(); } catch (error) { alert(error.message); } }
  el('source').onchange = () => guard(select);
  el('mark-start').onclick = () => { el('start').value = el('video').currentTime.toFixed(3); };
  el('mark-end').onclick = () => { el('end').value = el('video').currentTime.toFixed(3); };
  el('add').onclick = () => guard(() => {
    const proposed = [...entry().intervals, {start_seconds: Number(el('start').value), end_seconds: Number(el('end').value), action: el('action').value, evidence: el('evidence').value.trim()}].sort((a, b) => a.start_seconds - b.start_seconds);
    validateIntervals(proposed, selected.duration);
    entry().intervals = proposed; entry().review_status = 'pending'; persist(); render();
  });
  el('reviewer').oninput = () => { entry().reviewer = el('reviewer').value; if (!entry().reviewer.trim()) entry().review_status = 'pending'; persist(); };
  el('reviewed').onclick = () => guard(() => {
    if (!el('reviewer').value.trim() || !entry().intervals.length) throw Error('Cần người duyệt và ít nhất một đoạn.');
    entry().reviewer = el('reviewer').value.trim(); entry().review_status = 'reviewed'; persist(); render();
  });
  el('pending').onclick = () => { entry().review_status = 'pending'; persist(); render(); };
  el('download').onclick = () => {
    const link = document.createElement('a'), url = URL.createObjectURL(new Blob([JSON.stringify(labels, null, 2)], {type: 'application/json'}));
    link.href = url; link.download = 'assault_annotations.reviewed.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  el('import').onchange = async () => {
    try { labels = restore(pack.annotations, JSON.parse(await el('import').files[0].text()), pack.manifest.sources); persist(); render(); } catch (error) { alert(error.message); }
  };
  fetch('review.json', {cache: 'no-store'}).then(response => { if (!response.ok) throw Error('Không tải được bộ duyệt'); return response.json(); }).then(value => {
    pack = value; labels = JSON.parse(JSON.stringify(pack.annotations)); key = 'fightwatch.assault.' + pack.manifest.manifest_sha256;
    try { const saved = localStorage.getItem(key); if (saved) labels = restore(pack.annotations, JSON.parse(saved), pack.manifest.sources); } catch (error) { alert('Không khôi phục nhãn đã lưu: ' + error.message); }
    for (const [index, row] of pack.manifest.sources.entries()) { const option = document.createElement('option'); option.value = index; option.textContent = row.video; el('source').append(option); }
    persist(); select();
  }).catch(error => { el('message').textContent = error.message + ' · Mở bằng server HTTP local.'; for (const button of document.querySelectorAll('button')) button.disabled = true; });
})(typeof window === 'undefined' ? globalThis : window);
