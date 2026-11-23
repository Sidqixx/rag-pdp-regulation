const $ = s => document.querySelector(s);
const log = $('#log'), welcome = $('#welcome'), composer = $('#composer'), q = $('#q'), send = $('#send');
let history = [], n = 0, busy = false, controller = null;

const esc = s => s.replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

// Citation parsing mirrors ragpdp/answer.py (CITE, split_groups) so the UI flags exactly what the checker flags.
const DOC = String.raw`(UU(?: No\.?)? 27(?:\/| Tahun )2022|PP(?: No\.?)? 33(?:\/| Tahun )2026)`;
const DETAIL = String.raw`(?: [Aa]yat \(\d+\)(?:(?:,| dan) (?:[Aa]yat )?\(\d+\))*)?(?: [Hh]uruf [a-z])?`;
const CITE = String.raw`\[?${DOC} Pasal (\d+)${DETAIL}\]?|Pasal (\d+)${DETAIL} ${DOC}`;
const citeLabel = (d1, n1, n2, d2) => `${(d1 || d2).startsWith('UU') ? 'UU 27/2022' : 'PP 33/2026'} Pasal ${n1 || n2}`;
const cidOf = (id, label) => `${id}-${label.replace(/\W/g, '')}`;
const splitGroups = t => t.replace(/\[(UU 27\/2022|PP 33\/2026) (Pasal [^\[\]]+)\]/g, (_, doc, inner) => {
  const parts = [];
  for (const p of inner.split(/,\s*|\s+dan\s+/)) {
    if (p.startsWith('Pasal ') || !parts.length) parts.push(p);
    else parts[parts.length - 1] += ' dan ' + p;  // "ayat (2)" belongs to the previous pasal
  }
  return parts.map(p => `[${doc} ${p}]`).join(' ');
});

// Minimal markdown for LLM answers: paragraphs, headings, bullet/numbered lists, bold, italic, code.
// Input is already HTML-escaped, so nothing the model writes can inject markup.
const inline = s => s
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/(^|[\s(])\*(?!\s)([^*]+?)\*(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>')
  .replace(/`([^`]+)`/g, '<code>$1</code>');
function md(src) {
  let html = '', list = null, para = [];
  const flush = () => { if (para.length) html += `<p>${para.map(inline).join('<br>')}</p>`; para = []; };
  const close = () => { if (list) html += `</${list}>`; list = null; };
  for (const line of src.split('\n')) {
    let m;
    if (!line.trim()) { flush(); close(); }
    else if ((m = line.match(/^#{1,4}\s+(.*)/))) { flush(); close(); html += `<h3>${inline(m[1])}</h3>`; }
    else if ((m = line.match(/^(\s*)([*\-•]|(\d+)[.)])\s+(.*)/))) {
      flush();
      const type = m[3] ? 'ol' : 'ul';
      if (list !== type) { close(); html += `<${type}>`; list = type; }
      const level = Math.min(Math.floor(m[1].length / 2), 2);
      html += `<li class="lv${level}"${m[3] ? ` value="${m[3]}"` : ''}>${inline(m[4])}</li>`;
    } else { close(); para.push(line.trim()); }
  }
  flush(); close();
  return html;
}

function render(text, sources, id) {
  const ok = new Set(sources.map(s => s.label)), chips = [];
  const marked = esc(splitGroups(text)).replace(new RegExp(CITE, 'g'), (m, d1, n1, n2, d2) => {
    const l = citeLabel(d1, n1, n2, d2), shown = m.replace(/^\[|\]$/g, '');
    chips.push(ok.has(l)
      ? `<a class="cite" href="#${cidOf(id, l)}">${shown}</a>`
      : `<span class="cite bad" title="Pasal ini tidak ada di sumber yang diambil, jangan dipercaya">${shown}</span>`);
    return `\u0001${chips.length - 1}\u0002`;  // placeholder so markdown never touches chip HTML
  });
  return md(marked).replace(/\u0001(\d+)\u0002/g, (_, i) => chips[i]);
}

function sourcesHtml(text, sources, id) {
  const item = s => `<details id="${cidOf(id, s.label)}"><summary><span>${esc(s.label)} <span class="bab">· ${esc(s.bab)}</span></span></summary>` +
    `<div class="pasal">${esc(s.text)}</div>` +
    (s.penjelasan ? `<div class="pasal pj">Penjelasan:\n${esc(s.penjelasan)}</div>` : '') + '</details>';
  const cited = new Set([...splitGroups(text).matchAll(new RegExp(CITE, 'g'))].map(m => citeLabel(m[1], m[2], m[3], m[4])));
  const used = sources.filter(s => cited.has(s.label)), other = sources.filter(s => !cited.has(s.label));
  return '<div class="sources">' +
    (used.length ? '<strong>Sumber</strong>' + used.map(item).join('') : '') +
    (other.length ? `<details><summary>Pasal lain yang ikut dibaca (${other.length})</summary>${other.map(item).join('')}</details>` : '') +
    '</div>';
}

// Follow the stream only if the reader is already near the bottom.
const nearBottom = () => innerHeight + scrollY >= document.documentElement.scrollHeight - 160;
const toBottom = () => scrollTo({top: document.documentElement.scrollHeight});

function addUser(text) {
  const el = document.createElement('div');
  el.className = 'msg user';
  el.textContent = text;
  log.append(el);
}

function addBot() {
  const el = document.createElement('article');
  el.className = 'msg bot';
  el.innerHTML = '<span class="mark sm" aria-hidden="true"></span><div>' +
    '<p class="status"><span class="label">Mencari pasal</span><span class="dots"></span></p><div class="answer"></div>' +
    '<div class="actions" hidden><button type="button" class="btn ghost copy">' +
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>' +
    '<span>Salin</span></button></div></div>';
  log.append(el);
  return {status: el.querySelector('.status'), label: el.querySelector('.status .label'), answer: el.querySelector('.answer'), actions: el.querySelector('.actions')};
}

function updateSend() {
  const state = busy ? 'stop' : q.value.trim() ? 'send' : 'idle';
  send.dataset.state = state;
  send.disabled = state === 'idle';
  send.setAttribute('aria-label', busy ? 'Hentikan jawaban' : 'Kirim');
}

function autosize() {
  q.style.height = '0px';
  q.style.height = Math.min(Math.max(q.scrollHeight, 42), 168) + 'px';
  updateSend();
}

async function ask(question) {
  welcome.hidden = true;
  addUser(question);
  const id = 'm' + (n++), bot = addBot();
  toBottom();
  busy = true; controller = new AbortController(); updateSend();
  let sources = [], text = '', buf = '';
  try {
    const res = await fetch('/api/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, history}), signal: controller.signal,
    });
    if (res.status === 429) { bot.answer.textContent = (await res.json()).detail; return; }
    if (!res.ok) throw new Error(res.status);
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    for (let header = true;;) {
      const {value, done} = await reader.read();
      if (done) break;
      if (header) {
        buf += value;
        const nl = buf.indexOf('\n');
        if (nl < 0) continue;
        sources = JSON.parse(buf.slice(0, nl)).sources;
        text = buf.slice(nl + 1);
        header = false;
        bot.label.textContent = 'Menulis jawaban';
      } else text += value;
      if (text) {
        const follow = nearBottom();
        bot.status.hidden = true;
        bot.answer.innerHTML = render(text, sources, id);
        if (follow) toBottom();
      }
    }
    bot.answer.innerHTML = render(text, sources, id) + sourcesHtml(text, sources, id);
    bot.actions.dataset.text = text;
    bot.actions.hidden = false;
    history.push({role: 'user', content: question}, {role: 'assistant', content: text});
  } catch (e) {
    if (e.name === 'AbortError') {
      bot.answer.innerHTML = render(text, sources, id) + '<p class="stopped">Jawaban dihentikan.</p>' +
        (sources.length ? sourcesHtml(text, sources, id) : '');
    } else {
      bot.answer.textContent = 'Server belum bisa menjawab. Kirim ulang pertanyaan sebentar lagi.';
    }
  } finally {
    bot.status.hidden = true;
    busy = false; controller = null; updateSend();
  }
}

function submit() {
  const question = q.value.trim();
  if (!question || busy) return;
  q.value = '';
  autosize();
  ask(question);
}

q.addEventListener('input', autosize);
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
});
composer.addEventListener('submit', e => { e.preventDefault(); busy ? controller.abort() : submit(); });

document.querySelectorAll('#examples button').forEach(b => b.addEventListener('click', () => ask(b.textContent)));
log.addEventListener('click', async e => {
  const copy = e.target.closest('.copy');
  if (copy) {
    await navigator.clipboard.writeText(copy.parentElement.dataset.text);
    const label = copy.querySelector('span');
    label.textContent = 'Tersalin';
    setTimeout(() => { label.textContent = 'Salin'; }, 1500);
    return;
  }
  const a = e.target.closest('a.cite');
  if (!a) return;
  const d = document.querySelector(a.getAttribute('href'));
  if (d) d.open = true;
});
$('#reset').addEventListener('click', () => {
  controller?.abort();
  history = []; log.innerHTML = ''; welcome.hidden = false;
  scrollTo({top: 0});
});
const about = $('#about');
$('#about-open').addEventListener('click', () => about.showModal());
$('#about-close').addEventListener('click', () => about.close());
about.addEventListener('click', e => { if (e.target === about) about.close(); });
