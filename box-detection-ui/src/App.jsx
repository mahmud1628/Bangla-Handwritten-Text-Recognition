import { useEffect, useMemo, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_BOX_API_URL || 'http://127.0.0.1:8001'
const REQUEST_TIMEOUT_MS = 180000

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const previewUrl = useMemo(() => {
    if (!file) return ''
    return URL.createObjectURL(file)
  }, [file])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  async function handleSubmit(event) {
    event.preventDefault()

    if (!file) {
      setError('Please select an image first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

    try {
      const response = await fetch(`${API_BASE_URL}/detect-boxes-json`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      const payload = await response.json()

      if (!response.ok) {
        throw new Error(payload.detail || 'Box detection request failed.')
      }

      setResult(payload)
    } catch (requestError) {
      if (requestError?.name === 'AbortError') {
        setError('Request timed out. Please try a smaller or clearer image.')
      } else if (requestError instanceof TypeError) {
        setError('Could not reach backend API. Check backend URL and server status.')
      } else {
        setError(requestError.message || 'Unexpected API error occurred.')
      }
    } finally {
      clearTimeout(timeoutId)
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">Bangla Handwritten Pipeline</p>
        <h1>Box Detection UI</h1>
        <p>
          Upload a page image and view each detected handwritten box crop directly in the browser.
        </p>
      </header>

      <section className="panel">
        <form onSubmit={handleSubmit} className="form-grid">
          <label htmlFor="image-file">Input Image</label>
          <input
            id="image-file"
            type="file"
            accept="image/*"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />

          <button type="submit" disabled={loading}>
            {loading ? 'Detecting...' : 'Detect Boxes'}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {previewUrl && (
          <div className="preview-wrap">
            <h2>Input Preview</h2>
            <img src={previewUrl} alt="Input preview" className="preview-image" />
          </div>
        )}
      </section>

      {result && (
        <section className="panel result-panel">
          <div className="result-header">
            <h2>Detected Crops</h2>
            <span className="badge">{result.count} found</span>
          </div>

          {result.count === 0 ? (
            <p>No boxes detected. Try a cleaner image with stronger box outlines.</p>
          ) : (
            <div className="grid">
              {result.crops.map((crop, index) => {
                const boxMeta = result.boxes[index]
                return (
                  <article className="crop-card" key={crop.id}>
                    <img src={crop.image} alt={`Detected box ${crop.id}`} loading="lazy" />
                    <div className="meta">
                      <strong>Box {crop.id}</strong>
                      <span>conf: {boxMeta?.confidence ?? 'n/a'}</span>
                      <span>
                        size: {boxMeta?.bbox?.w ?? '-'} x {boxMeta?.bbox?.h ?? '-'}
                      </span>
                    </div>
                  </article>
                )
              })}
            </div>
          )}
        </section>
      )}
    </main>
  )
}

export default App
