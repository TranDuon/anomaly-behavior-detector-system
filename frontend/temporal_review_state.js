(function (root) {
  const actions = new Set(['fight', 'hug', 'handshake_greet', 'highfive_clap', 'gesture', 'solo_motion', 'normal', 'uncertain']);

  function restoreAnnotations(template, saved) {
    const restored = JSON.parse(JSON.stringify(template));
    if (!saved || saved.schema_version !== template.schema_version || saved.source_manifest_sha256 !== template.source_manifest_sha256 || !Array.isArray(saved.sources)) return restored;
    const entries = new Map();
    for (const entry of saved.sources) {
      if (!entry || entries.has(entry.video)) return restored;
      entries.set(entry.video, entry);
    }
    restored.sources = restored.sources.map(source => {
      const entry = entries.get(source.video);
      if (!entry || entry.sha256 !== source.sha256 || entry.split !== source.split || !Array.isArray(entry.intervals)) return source;
      let end = 0;
      for (const interval of entry.intervals) {
        if (!interval || !actions.has(interval.action) || !Number.isFinite(interval.start_seconds) || !Number.isFinite(interval.end_seconds) || interval.start_seconds < end || interval.end_seconds <= interval.start_seconds) return source;
        end = interval.end_seconds;
      }
      const result = {...source, intervals: entry.intervals.map(interval => ({...interval})), reviewer: typeof entry.reviewer === 'string' ? entry.reviewer : '', review_method: entry.review_method || ''};
      result.review_status = entry.review_status === 'approved' && result.reviewer.trim() && ['human', 'ai_assisted'].includes(result.review_method) && result.intervals.length ? 'approved' : 'pending';
      return result;
    });
    return restored;
  }

  const api = {restoreAnnotations};
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.FightTemporalReview = api;
})(typeof window === 'undefined' ? this : window);
