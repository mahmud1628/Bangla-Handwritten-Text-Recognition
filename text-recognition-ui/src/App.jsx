import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.DEV ? 'http://127.0.0.1:8000/recognize' : '/api/recognize'
const REQUEST_TIMEOUT_SECONDS = 300

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
      setError('Please choose an image first.')
      return
    }

    setError('')
    setResult(null)
    setLoading(true)
    let timeoutId

    try {
      const formData = new FormData()
      formData.append('file', file)

      const controller = new AbortController()
      timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_SECONDS * 1000)

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      const payload = await response.json()

      if (!response.ok) {
        throw new Error(payload.detail || 'Recognition failed.')
      }

      setResult(payload)
    } catch (submitError) {
      if (submitError?.name === 'AbortError') {
        setError('Request timed out after 5 minutes. Please try again.')
      } else if (submitError instanceof TypeError) {
        setError('Failed to fetch backend API. Check backend URL, server status, and CORS settings.')
      } else {
        setError(submitError.message || 'Failed to call backend API.')
      }
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <h1>Bangla Handwritten Text Recognition</h1>
      <p className="subtitle">
        Upload an image and call the backend API to get line count, word count, and full text.
      </p>

      <section className="card">
        <form className="form-grid" onSubmit={handleSubmit}>
          <div>
            <label htmlFor="file-input">Document Image</label>
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Recognizing...' : 'Recognize Text'}
          </button>
        </form>

        <p className="subtitle">Large models may take 1-2 minutes or more per request.</p>

        {error && <p className="error">{error}</p>}

        {previewUrl && (
          <div className="preview">
            <label>Selected Image Preview</label>
            <img src={previewUrl} alt="Selected document preview" />
          </div>
        )}

        {result && (
          <div className="results">
            <div className="stat-grid">
              <div className="stat">
                <div className="label">Line Count</div>
                <div className="value">{result.line_count}</div>
              </div>
              <div className="stat">
                <div className="label">Word Count</div>
                <div className="value">{result.word_count}</div>
              </div>
            </div>

            <div>
              <label htmlFor="full-text">Detected Full Text</label>
              <textarea id="full-text" value={result.full_text || ''} readOnly />
            </div>
          </div>
        )}
      </section>
    </main>
  )
}

export default App
