const BASE_URL = 'http://localhost:8000'
// TTS runs as a separate process (tts_app.py, its own conda env) since it
// needs transformers, which the main dashboard's env deliberately lacks.
const TTS_API_BASE = 'http://localhost:8001'

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new Error(detail)
  }
  return res.json()
}

export function predictMask(sentence) {
  return request('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentence }),
  })
}

export function getPerplexity() {
  return request('/api/perplexity')
}

export function generateClm(prompt, numNewTokens) {
  return request('/api/clm/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, num_new_tokens: numNewTokens }),
  })
}

export function translate(text, direction) {
  return request('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, direction }),
  })
}

async function synthesizeAt(path, text) {
  const res = await fetch(`${TTS_API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new Error(detail)
  }
  return res.blob()
}

export function synthesize(text) {
  return synthesizeAt('/api/tts/synthesize', text)
}

// facebook/mms-tts-mai -- the pretrained checkpoint fine-tuned into
// checkpoint-5500 -- for a before/after fine-tuning comparison.
export function synthesizeBase(text) {
  return synthesizeAt('/api/tts/synthesize-base', text)
}
