import { useState, useRef, useEffect, useCallback, DragEvent } from 'react'
import { Mic, Upload, AlertTriangle, Loader2, HelpCircle } from 'lucide-react'
import { useRecorder } from '../hooks/useRecorder'
import { analizar } from '../api'
import type { ResultadoAnalisis } from '../types'

const urgencyBanner: Record<string, string> = {
  ALTA: 'bg-red-500 text-white',
  MEDIA: 'bg-amber-400 text-white',
  BAJA: 'bg-emerald-500 text-white',
}

const urgencyLabel: Record<string, string> = {
  ALTA: '⚠ URGENTE',
  MEDIA: 'ATENCIÓN',
  BAJA: 'NORMAL',
}

const confianzaBadge: Record<string, string> = {
  ALTO: 'bg-emerald-100 text-emerald-700 border-emerald-200',
  MEDIO: 'bg-amber-100 text-amber-700 border-amber-200',
  BAJO: 'bg-red-100 text-red-700 border-red-200',
}

const confianzaLabel: Record<string, string> = {
  ALTO: 'Confianza alta',
  MEDIO: 'Confianza media',
  BAJO: 'Confianza baja',
}

export function AnalizarTab() {
  const { isRecording, audioBlob, error, startRecording, stopRecording, reset } =
    useRecorder()
  const [resultado, setResultado] = useState<ResultadoAnalisis | null>(null)
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const blobProcessed = useRef(false)

  const audioRef = useRef<HTMLAudioElement | null>(null)

  const handleBlob = useCallback(async (blob: Blob) => {
    setLoading(true)
    setApiError(null)
    try {
      const res = await analizar(blob)
      setResultado(res)
      if (res.audio_id && !res.no_reconocido) {
        const audio = new Audio(`/audio_respuesta/${res.audio_id}`)
        audioRef.current = audio
        audio.play().catch(() => {})
      }
    } catch {
      setApiError('Error al analizar el audio. Intenta de nuevo.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (audioBlob && !blobProcessed.current) {
      blobProcessed.current = true
      handleBlob(audioBlob).then(() => {
        reset()
        blobProcessed.current = false
      })
    }
  }, [audioBlob, handleBlob, reset])

  const handleFile = (file: File) => {
    const allowed = ['audio/webm', 'audio/mp4', 'audio/wav', 'audio/ogg', 'video/mp4']
    if (!allowed.includes(file.type) && !file.name.match(/\.(webm|mp4|wav|ogg)$/i)) {
      setApiError('Formato no válido. Usa webm, mp4, wav u ogg.')
      return
    }
    handleBlob(file)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => setIsDragging(false)

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`flex flex-col items-center justify-center gap-3 p-8 rounded-2xl border-2 border-dashed cursor-pointer transition-all ${
            isDragging
              ? 'border-violet-500 bg-violet-50'
              : 'border-slate-300 bg-white hover:border-violet-400 hover:bg-violet-50/40'
          }`}
        >
          <Upload size={28} className="text-slate-400" />
          <div className="text-center">
            <p className="text-sm font-medium text-slate-700">
              Arrastra un archivo de audio
            </p>
            <p className="text-xs text-slate-400 mt-1">webm · mp4 · wav · ogg</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".webm,.mp4,.wav,.ogg,audio/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
              e.target.value = ''
            }}
          />
        </div>

        <div className="flex flex-col items-center justify-center gap-4 p-8 rounded-2xl border border-slate-200 bg-white">
          <p className="text-sm font-medium text-slate-600">
            Grabar desde micrófono
          </p>
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={loading}
            className={`w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-lg focus:outline-none focus:ring-4 focus:ring-violet-300 ${
              loading
                ? 'bg-slate-200 cursor-not-allowed'
                : isRecording
                  ? 'bg-red-500 animate-pulse shadow-red-300 hover:bg-red-600'
                  : 'bg-violet-600 hover:bg-violet-700 shadow-violet-300'
            }`}
          >
            {loading ? (
              <Loader2 size={36} className="text-slate-400 animate-spin" />
            ) : (
              <Mic
                size={36}
                className={isRecording ? 'text-white' : 'text-white'}
              />
            )}
          </button>
          <p className="text-xs text-slate-400">
            {isRecording ? 'Grabando... (se detiene a los 4s)' : 'Pulsa para grabar'}
          </p>
        </div>
      </div>

      {(error || apiError) && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
          <AlertTriangle size={16} />
          {error || apiError}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center gap-3 p-8 rounded-2xl bg-white border border-slate-200">
          <Loader2 size={24} className="text-violet-600 animate-spin" />
          <span className="text-slate-600 font-medium">Analizando audio...</span>
        </div>
      )}

      {resultado && !loading && resultado.no_reconocido && (
        <div className="rounded-2xl overflow-hidden border border-slate-200 shadow-sm bg-white">
          <div className="px-6 py-4 flex items-center gap-3 bg-slate-100">
            <HelpCircle size={20} className="text-slate-500" />
            <span className="font-bold text-lg tracking-wide text-slate-600">No reconocido</span>
          </div>
          <div className="p-6 space-y-3">
            <p className="text-slate-600 text-sm">
              El audio no coincide con ninguna frase calibrada. Intenta grabar más cerca o calibra más muestras.
            </p>
            {resultado.distancia_dtw !== null && resultado.distancia_dtw !== undefined && (
              <p className="text-xs text-slate-400 font-mono">
                Distancia DTW: {resultado.distancia_dtw.toFixed(3)} (umbral: 0.800)
              </p>
            )}
          </div>
        </div>
      )}

      {resultado && !loading && !resultado.no_reconocido && (
        <div className="rounded-2xl overflow-hidden border border-slate-200 shadow-sm bg-white">
          <div
            className={`px-6 py-4 flex items-center gap-3 ${
              urgencyBanner[resultado.urgencia] ?? 'bg-slate-500 text-white'
            }`}
          >
            <span className="font-bold text-lg tracking-wide">
              {urgencyLabel[resultado.urgencia] ?? resultado.urgencia}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
            <div className="space-y-4">
              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-1">
                  Frase detectada
                </p>
                <p className="text-2xl font-bold text-slate-800">
                  {resultado.frase_detectada}
                </p>
              </div>

              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-1">
                  Significado
                </p>
                <p className="text-slate-700">{resultado.significado}</p>
              </div>

              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-1">
                  Respuesta del cuidador
                </p>
                <div className="bg-violet-50 border border-violet-100 rounded-xl px-4 py-3">
                  <p className="text-violet-800 text-sm leading-relaxed">
                    {resultado.respuesta}
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-2">
                  Confianza
                </p>
                <span
                  className={`inline-flex items-center px-3 py-1.5 rounded-full border text-sm font-semibold ${
                    confianzaBadge[resultado.confianza] ??
                    'bg-slate-100 text-slate-700 border-slate-200'
                  }`}
                >
                  {confianzaLabel[resultado.confianza] ?? resultado.confianza}
                </span>
              </div>

              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-1">
                  Distancia DTW
                </p>
                <p className="text-2xl font-mono font-bold text-slate-700">
                  {resultado.distancia_dtw.toFixed(2)}
                </p>
              </div>

              {resultado.segunda_opcion && (
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-2">
                    Segunda opción
                  </p>
                  <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 space-y-1">
                    <p className="font-semibold text-slate-700 text-sm">
                      {resultado.segunda_opcion}
                    </p>
                    {resultado.dist_segunda !== null && (
                      <p className="text-xs text-slate-400 font-mono">
                        DTW: {resultado.dist_segunda.toFixed(2)}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
