// CloudScribe Browser MVP. No media upload: the user-selected File stays in this tab.
// External dependencies and the Whisper model are downloaded by the browser.
import { Input, ALL_FORMATS, BlobSource, AudioSampleSink } from 'https://cdn.jsdelivr.net/npm/mediabunny@1.62.1/+esm';
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

const $ = (id) => document.getElementById(id);
let file = null;
let transcriber = null;
let transcriberKey = '';
let cancelled = false;
let busy = false;
let segments = [];

// Loading remote model weights is unavoidable on the first run of this CDN-based prototype.
env.allowLocalModels = false;

const formatBytes = (bytes) => (bytes / 1024 / 1024).toFixed(1) + ' МБ';
const formatTime = (seconds) => {
  const n = Math.floor(Math.max(0, seconds));
  const h = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), s = n % 60;
  return h ? `${h} ч ${m} мин ${s} с` : `${m} мин ${s} с`;
};
const showMessage = (msg, error = false) => {
  $('message').textContent = msg;
  $('message').className = 'message ' + (error ? 'error' : 'ok');
};
const status = (message, tag = 'Обработка', css = 'running') => {
  $('stage').textContent = message;
  $('status-tag').textContent = tag;
  $('status-tag').className = 'status ' + css;
};
const progress = (fraction) => {
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  $('progress-fill').style.width = pct + '%';
  $('progress-track').setAttribute('aria-valuenow', String(pct));
  $('percent').textContent = pct + '%';
};
const setBusy = (value) => {
  busy = value;
  $('start').disabled = value || !file;
  $('stop').hidden = !value;
  $('file-input').disabled = value;
};
function chooseFile(nextFile) {
  if (busy || !nextFile) return;
  file = nextFile;
  segments = [];
  $('file-label').textContent = nextFile.name;
  $('file-meta').textContent = formatBytes(nextFile.size);
  $('transcript').value = '';
  $('transcript').disabled = true;
  for (const id of ['save-txt', 'save-srt', 'save-json']) $(id).disabled = true;
  $('details').textContent = `Выбрано: ${nextFile.name} (${formatBytes(nextFile.size)})`;
  status('Файл выбран', 'Готов к запуску', '');
  progress(0);
  showMessage('');
  setBusy(false);
}
$('drop-area').addEventListener('click', () => { if (!busy) $('file-input').click(); });
$('drop-area').addEventListener('keydown', (e) => { if (!busy && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); $('file-input').click(); } });
$('file-input').addEventListener('change', e => chooseFile(e.target.files?.[0]));
$('drop-area').addEventListener('dragover', e => {e.preventDefault();$('drop-area').classList.add('drag');});
$('drop-area').addEventListener('dragleave', () => $('drop-area').classList.remove('drag'));
$('drop-area').addEventListener('drop', e => {e.preventDefault();$('drop-area').classList.remove('drag');chooseFile(e.dataTransfer.files?.[0]);});
$('stop').addEventListener('click', () => {cancelled = true;showMessage('Остановка после завершения текущего фрагмента…');});

function languageOption() {
  if ($('language').value === 'auto') return {};
  return {language: $('language').value === 'ru' ? 'russian' : 'english'};
}
async function loadModel() {
  const device = $('engine').value === 'auto' ? (navigator.gpu ? 'webgpu' : 'wasm') : $('engine').value;
  if (device === 'webgpu' && !navigator.gpu) throw Error('Этот браузер не поддерживает WebGPU. Выберите «Автоматически» или CPU.');
  const model = $('quality').value === 'fast' ? 'onnx-community/whisper-tiny' : 'onnx-community/whisper-small';
  const key = `${model}::${device}`;
  if (transcriber && transcriberKey === key) return transcriber;
  if (transcriber?.dispose) {await transcriber.dispose();transcriber = null;}
  status('Загрузка модели Whisper…');
  showMessage('Модель загружается с Hugging Face. При первом запуске это может потребовать времени и значительного объёма трафика.');
  let next;
  try {
    next = await pipeline('automatic-speech-recognition', model, {
      device,
      dtype: device === 'webgpu' ? {encoder_model: 'fp32', decoder_model_merged: 'q4'} : 'q8',
      progress_callback: (event) => {
        if (event.status === 'progress' && typeof event.progress === 'number')
          status(`Загрузка модели: ${Math.round(event.progress)}%`);
      },
    });
  } catch (err) {
    if (device === 'webgpu' && $('engine').value === 'auto') {
      showMessage('WebGPU не смог загрузить модель. Повторная попытка на CPU…');
      next = await pipeline('automatic-speech-recognition', model, {device: 'wasm', dtype: 'q8'});
    } else {throw err;}
  }
  transcriber = next;
  transcriberKey = key;
  return transcriber;
}

// Decode only a short time range; never load a multi-hour video into a giant AudioBuffer.
// Linear sample-rate conversion to 16 kHz is adequate for an initial ASR prototype.
async function audioWindow(sink, from, to) {
  const sampleRate = 16000;
  const output = new Float32Array(Math.ceil((to - from) * sampleRate));
  for await (const sample of sink.samples(from, to)) {
    try {
      const buffer = sample.toAudioBuffer();
      const sourceRate = buffer.sampleRate;
      const channels = Array.from({length: buffer.numberOfChannels}, (_, i) => buffer.getChannelData(i));
      const first = Math.max(0, Math.ceil((sample.timestamp - from) * sampleRate));
      const last = Math.min(output.length, Math.ceil((sample.timestamp + sample.duration - from) * sampleRate));
      for (let j = first; j < last; j++) {
        const pos = (from + j / sampleRate - sample.timestamp) * sourceRate;
        const low = Math.min(channels[0].length - 1, Math.max(0, Math.floor(pos)));
        const high = Math.min(channels[0].length - 1, low + 1);
        const a = Math.max(0, Math.min(1, pos - low));
        let mono = 0;
        for (const channel of channels) mono += channel[low] * (1 - a) + channel[high] * a;
        output[j] = mono / channels.length;
      }
    } finally {sample.close();}
    if (cancelled) break;
  }
  return output;
}

function cleanSegments(output, offset, clipStart, clipEnd) {
  const chunks = Array.isArray(output?.chunks) ? output.chunks : [];
  if (chunks.length === 0) {
    const text = String(output?.text || '').trim();
    return text ? [{start: clipStart, end: clipEnd, text}] : [];
  }
  return chunks.map(c => {
    const begin = Math.max(clipStart, offset + Number(c.timestamp?.[0] ?? 0));
    const end = Math.min(clipEnd, offset + Number(c.timestamp?.[1] ?? (clipEnd - offset)));
    return {start: begin, end: Math.max(begin + .01, end), text: String(c.text || '').trim()};
  }).filter(s => s.text && s.start < clipEnd && s.end > clipStart);
}
function updateText() {
  $('transcript').value = segments.map(s => s.text).join('\n');
  $('transcript').disabled = false;
  for (const id of ['save-txt','save-srt','save-json']) $(id).disabled = !segments.length;
}
async function run() {
  if (!file || busy) return;
  setBusy(true);
  cancelled = false;
  segments = [];
  $('transcript').value = '';
  $('transcript').disabled = true;
  progress(0);
  let input;
  try {
    status('Анализ аудиодорожки…');
    input = new Input({formats: ALL_FORMATS, source: new BlobSource(file)});
    const track = await input.getPrimaryAudioTrack();
    if (!track) throw Error('В этом файле нет аудиодорожки.');
    if (!await track.canDecode()) throw Error('Браузер не может декодировать этот аудиокодек. Попробуйте MP4 с AAC или WAV/MP3.');
    const duration = await track.computeDuration();
    if (!Number.isFinite(duration) || duration <= 0) throw Error('Не удалось определить длительность аудио.');
    $('details').textContent = `${file.name} · ${formatTime(duration)} · ${formatBytes(file.size)}`;
    const sink = new AudioSampleSink(track);
    const pipe = await loadModel();
    if (cancelled) return;
    const coreSize = 24;
    const chunkCount = Math.ceil(duration / coreSize);
    for (let i = 0; i < chunkCount && !cancelled; i++) {
      const coreStart = i * coreSize;
      const coreEnd = Math.min(duration, (i + 1) * coreSize);
      const from = Math.max(0, coreStart - 1);
      const to = Math.min(duration, coreEnd + 1);
      status(`Извлечение аудио: фрагмент ${i + 1} из ${chunkCount}`);
      const pcm = await audioWindow(sink, from, to);
      if (cancelled) break;
      status(`Распознавание: фрагмент ${i + 1} из ${chunkCount}`);
      const result = await pipe(pcm, {return_timestamps: true, task: 'transcribe', ...languageOption()});
      segments.push(...cleanSegments(result, from, coreStart, coreEnd));
      updateText();
      progress((i + 1) / chunkCount);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (cancelled) {
      status('Остановлено — результаты уже распознанных фрагментов сохранены на странице.', 'Остановлено', '');
    } else {
      status(`Расшифровка завершена. Сегментов: ${segments.length}.`, 'Готово', 'done');
      progress(1);
      showMessage('Готово. При необходимости исправьте текст и сохраните TXT, SRT или JSON.');
    }
  } catch (err) {
    console.error(err);
    status('Ошибка', 'Ошибка', 'error');
    showMessage(err?.message || String(err), true);
  } finally {
    input?.dispose?.();
    setBusy(false);
  }
}
$('start').addEventListener('click', run);

function timestamp(seconds, comma) {
  const ms = Math.round(Math.max(0, seconds) * 1000);
  const h = Math.floor(ms / 3600000), m = Math.floor(ms % 3600000 / 60000);
  const s = Math.floor(ms % 60000 / 1000), remainder = ms % 1000;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}${comma?',':'.'}${String(remainder).padStart(3,'0')}`;
}
function download(format) {
  if (!segments.length || !file) return;
  let content, mime;
  if (format === 'txt') {content = '\ufeff' + $('transcript').value;mime = 'text/plain;charset=utf-8';}
  if (format === 'srt') {
    content = segments.map((s,i) => `${i+1}\n${timestamp(s.start,true)} --> ${timestamp(s.end,true)}\n${s.text}`).join('\n\n') + '\n';
    mime = 'text/plain;charset=utf-8';
  }
  if (format === 'json') {content = JSON.stringify(segments, null, 2);mime = 'application/json;charset=utf-8';}
  const url = URL.createObjectURL(new Blob([content], {type:mime}));
  const a = document.createElement('a');a.href = url;a.download = file.name.replace(/\.[^.]*$/, '') + '.' + format;
  document.body.appendChild(a);a.click();a.remove();setTimeout(() => URL.revokeObjectURL(url), 6000);
}
$('save-txt').addEventListener('click', () => download('txt'));
$('save-srt').addEventListener('click', () => download('srt'));
$('save-json').addEventListener('click', () => download('json'));
