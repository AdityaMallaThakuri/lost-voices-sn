import { useEffect, useRef, useState } from 'react'
import { synthesize, synthesizeBase } from '../api.js'

// The sentence verified end-to-end during the backend smoke test.
const EXAMPLES = [
  { label: 'Verified test sentence', text: 'परमप्रभु यावे आ दाक्‍शो पा' },
  { label: 'Proper name (subject)', text: 'मिनु आ खिं लशा बाक्‍तक।' },
  { label: 'Negated verb in question', text: 'आफोमी शेंचा तौ बाक्‍बा ङा?' },
]

function connectionErrorMessage(err) {
  return err.message === 'Failed to fetch'
    ? 'Could not reach the TTS server at localhost:8001 — is tts_app.py running? (uvicorn tts_app:app --port 8001, in the tts_env conda env)'
    : err.message
}

function TtsPage() {
  const [text, setText] = useState(EXAMPLES[0].text)
  const [loading, setLoading] = useState(false)

  const [audioUrl, setAudioUrl] = useState(null)
  const [error, setError] = useState(null)
  const audioUrlRef = useRef(null)

  const [baseAudioUrl, setBaseAudioUrl] = useState(null)
  const [baseError, setBaseError] = useState(null)
  const baseAudioUrlRef = useRef(null)

  // Revoke previous blob URLs whenever new ones are created, and on unmount,
  // so synthesized audio doesn't leak memory across requests.
  useEffect(() => {
    return () => {
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
      if (baseAudioUrlRef.current) URL.revokeObjectURL(baseAudioUrlRef.current)
    }
  }, [])

  async function handleSynthesize(e) {
    e.preventDefault()
    setError(null)
    setBaseError(null)
    setLoading(true)

    const [ours, base] = await Promise.allSettled([synthesize(text), synthesizeBase(text)])

    if (ours.status === 'fulfilled') {
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
      const url = URL.createObjectURL(ours.value)
      audioUrlRef.current = url
      setAudioUrl(url)
    } else {
      setError(connectionErrorMessage(ours.reason))
      setAudioUrl(null)
    }

    if (base.status === 'fulfilled') {
      if (baseAudioUrlRef.current) URL.revokeObjectURL(baseAudioUrlRef.current)
      const url = URL.createObjectURL(base.value)
      baseAudioUrlRef.current = url
      setBaseAudioUrl(url)
    } else {
      setBaseError(connectionErrorMessage(base.reason))
      setBaseAudioUrl(null)
    }

    setLoading(false)
  }

  return (
    <section className="page-section">
      <h1 className="page-title">Sunuwar TTS</h1>
      <p className="page-subtitle">
        VITS speech synthesis · fine-tuned from checkpoint-5500.
      </p>

      <div className="example-chips">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            className="chip"
            onClick={() => setText(ex.text)}
          >
            {ex.label}
          </button>
        ))}
      </div>

      <form className="input-row" onSubmit={handleSynthesize}>
        <input
          type="text"
          className="text-input font-content"
          placeholder="e.g. परमप्रभु यावे आ दाक्‍शो पा"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit" className="btn-solid" disabled={loading}>
          {loading ? 'Synthesizing…' : 'Synthesize'}
        </button>
      </form>

      <div className="interlinear">
        <p className="interlinear-source">{text}</p>
        {loading && <p className="page-subtitle">Synthesizing audio, this can take a few seconds…</p>}

        <div style={{ marginTop: '12px' }}>
          <p className="field-label">Our fine-tuned model (checkpoint-5500)</p>
          {error && <p className="error-text">{error}</p>}
          {audioUrl && !error && <audio controls src={audioUrl} />}
        </div>

        <div style={{ marginTop: '12px' }}>
          <p className="field-label">Base model, before fine-tuning (facebook/mms-tts-mai)</p>
          {baseError && <p className="error-text">{baseError}</p>}
          {baseAudioUrl && !baseError && <audio controls src={baseAudioUrl} />}
        </div>
      </div>

      <p className="ceiling-note">
        Checkpoint-5500 — partial training (target: ~14,000 steps, per
        configs/tts.yaml) — audio
        quality reflects in-progress fine-tuning. The base model was never
        trained on Sunuwar (it's Maithili, chosen for its Devanagari character
        coverage) — its output shows what fine-tuning changed, not a native
        Sunuwar voice.
      </p>
    </section>
  )
}

export default TtsPage
