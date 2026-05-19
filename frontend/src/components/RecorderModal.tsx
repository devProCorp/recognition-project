import { useEffect, useRef, useState } from 'react'
import { X, Mic, CheckCircle, Upload, FileAudio, Loader2, Trash2 } from 'lucide-react'
import { useRecorder } from '../hooks/useRecorder'
import { calibrar, borrarCalibracion } from '../api'

type Mode = 'record' | 'upload'
type Status = 'idle' | 'recording' | 'processing' | 'saved' | 'error'

interface Props {
  frase: string
  muestras: number
  onClose: () => void
  onSampleAdded: (newCount: number) => void
  onDeleted?: () => void
}

export function RecorderModal({ frase, muestras, onClose, onSampleAdded, onDeleted }: Props) {
  const { isRecording, audioBlob, error: recError, startRecording, stopRecording, reset } =
    useRecorder()
  const [mode, setMode] = useState<Mode>('record')
  const [currentMuestras, setCurrentMuestras] = useState(muestras)
  const [status, setStatus] = useState<Status>('idle')
  const [statusMsg, setStatusMsg] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const [queueProgress, setQueueProgress] = useState<{ current: number; total: number } | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function handleDelete() {
    if (!confirmDelete) { setConfirmDelete(true); return }
    setDeleting(true)
    try {
      await borrarCalibracion(frase)
      setCurrentMuestras(0)
      onSampleAdded(0)
      onDeleted?.()
      onClose()
    } catch {
      setStatus('error')
      setStatusMsg('Error al borrar la calibración')
    } finally {
      setDeleting(false)
      setConfirmDelete(false)
    }
  }

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null
    if (isRecording) {
      setStatus('recording')
      setElapsed(0)
      interval = setInterval(() => setElapsed((p) => +(p + 0.1).toFixed(1)), 100)
    }
    return () => { if (interval) clearInterval(interval) }
  }, [isRecording])

  useEffect(() => {
    if (!audioBlob) return
    submitBlob(audioBlob)
  }, [audioBlob])

  async function submitBlob(blob: Blob): Promise<number> {
    setStatus('processing')
    setStatusMsg('Procesando...')
    try {
      const res = await calibrar(frase, blob)
      setCurrentMuestras(res.muestras)
      onSampleAdded(res.muestras)
      setStatus('saved')
      setStatusMsg(`¡Guardado! ${res.muestras} de 3`)
      reset()
      return res.muestras
    } catch {
      setStatus('error')
      setStatusMsg('Error al procesar el audio')
      reset()
      return -1
    }
  }

  async function handleFiles(files: FileList) {
    const validExtensions = /\.(mp4|m4a|ogg|webm|wav|aac|mp3|opus|wma|flac)$/i
    const audioFiles = Array.from(files).filter(
      (f) => f.type.startsWith('audio/') || f.type.startsWith('video/') || validExtensions.test(f.name)
    )
    if (!audioFiles.length) return

    setQueueProgress({ current: 0, total: audioFiles.length })

    for (let i = 0; i < audioFiles.length; i++) {
      setQueueProgress({ current: i + 1, total: audioFiles.length })
      setStatusMsg(`Procesando archivo ${i + 1} de ${audioFiles.length}…`)
      setStatus('processing')
      await submitBlob(audioFiles[i])
    }

    setQueueProgress(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  function switchMode(m: Mode) {
    if (isRecording) stopRecording()
    setMode(m)
    setStatus('idle')
    setStatusMsg('')
    setQueueProgress(null)
  }

  const recordStatusText =
    status === 'recording'
      ? `Grabando… ${elapsed.toFixed(1)}s`
      : statusMsg || (currentMuestras >= 3 ? 'Frase lista — puedes agregar más' : 'Toca para grabar')

  const uploadStatusText =
    statusMsg || 'Arrastra uno o varios archivos de audio o haz click para buscar'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="relative w-full max-w-sm rounded-2xl bg-white shadow-2xl p-6 mx-4">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 transition-colors"
        >
          <X size={20} />
        </button>

        <div className="flex items-start justify-between gap-2 pr-6 mb-1">
          <h2 className="text-xl font-bold text-slate-800 leading-tight">{frase}</h2>
          {currentMuestras > 0 && (
            <button
              onClick={handleDelete}
              disabled={deleting}
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors flex-shrink-0 mt-0.5 ${
                confirmDelete
                  ? 'bg-red-500 text-white hover:bg-red-600'
                  : 'text-slate-400 hover:text-red-500 hover:bg-red-50'
              }`}
            >
              {deleting ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
              {confirmDelete ? '¿Confirmar?' : 'Borrar'}
            </button>
          )}
        </div>
        <p className="text-sm text-slate-500 mb-4">Calibración de voz</p>

        {/* Sample dots */}
        <div className="flex justify-center gap-3 mb-5">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={`w-10 h-10 rounded-full border-2 flex items-center justify-center transition-all ${
                i < currentMuestras
                  ? 'bg-emerald-500 border-emerald-500'
                  : 'bg-white border-slate-300'
              }`}
            >
              {i < currentMuestras && <CheckCircle size={18} className="text-white" />}
            </div>
          ))}
        </div>

        {/* Mode tabs */}
        <div className="flex rounded-lg overflow-hidden border border-slate-200 mb-5">
          {(['record', 'upload'] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => switchMode(m)}
              className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium transition-colors ${
                mode === m ? 'bg-violet-600 text-white' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              {m === 'record' ? (
                <><Mic size={14} /> Grabar</>
              ) : (
                <><Upload size={14} /> Subir fragmento</>
              )}
            </button>
          ))}
        </div>

        {/* ── Record mode ── */}
        {mode === 'record' && (
          <div className="space-y-5">
            <p
              className={`text-center text-sm font-medium ${
                status === 'recording'
                  ? 'text-red-500'
                  : status === 'saved'
                    ? 'text-emerald-600'
                    : status === 'processing'
                      ? 'text-violet-600'
                      : status === 'error'
                        ? 'text-red-500'
                        : 'text-slate-500'
              }`}
            >
              {recordStatusText}
            </p>

            {recError && (
              <p className="text-center text-xs text-red-500">{recError}</p>
            )}

            <div className="flex justify-center">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={status === 'processing'}
                className={`w-20 h-20 rounded-full flex items-center justify-center transition-all shadow-lg focus:outline-none ${
                  status === 'processing'
                    ? 'bg-slate-200 cursor-not-allowed'
                    : isRecording
                      ? 'bg-red-500 animate-pulse shadow-red-300 hover:bg-red-600'
                      : 'bg-slate-200 hover:bg-slate-300'
                }`}
              >
                {status === 'processing' ? (
                  <Loader2 size={32} className="text-slate-400 animate-spin" />
                ) : (
                  <Mic size={32} className={isRecording ? 'text-white' : 'text-slate-600'} />
                )}
              </button>
            </div>
          </div>
        )}

        {/* ── Upload mode ── */}
        {mode === 'upload' && (
          <div className="space-y-3">
            <p
              className={`text-center text-sm font-medium ${
                status === 'saved'
                  ? 'text-emerald-600'
                  : status === 'processing'
                    ? 'text-violet-600'
                    : status === 'error'
                      ? 'text-red-500'
                      : 'text-slate-500'
              }`}
            >
              {uploadStatusText}
            </p>

            {/* Queue progress bar */}
            {queueProgress && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-slate-500">
                  <span>Procesando…</span>
                  <span>{queueProgress.current}/{queueProgress.total}</span>
                </div>
                <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-violet-500 rounded-full transition-all duration-300"
                    style={{ width: `${(queueProgress.current / queueProgress.total) * 100}%` }}
                  />
                </div>
              </div>
            )}

            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { e.preventDefault(); setIsDragging(false); e.dataTransfer.files && handleFiles(e.dataTransfer.files) }}
              onClick={() => status !== 'processing' && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center gap-2 transition-colors ${
                status === 'processing'
                  ? 'opacity-60 cursor-not-allowed border-slate-300'
                  : isDragging
                    ? 'border-violet-400 bg-violet-50 cursor-copy'
                    : 'border-slate-300 hover:border-violet-400 hover:bg-slate-50 cursor-pointer'
              }`}
            >
              {status === 'processing' ? (
                <Loader2 size={28} className="text-violet-400 animate-spin" />
              ) : (
                <FileAudio size={28} className={isDragging ? 'text-violet-500' : 'text-slate-400'} />
              )}
              <p className="text-sm text-slate-700 font-medium text-center">
                {isDragging ? 'Suelta aquí' : 'Arrastra audios aquí'}
              </p>
              <p className="text-xs text-slate-400 text-center">o haz click para buscar archivos</p>
              <p className="text-xs text-slate-300">mp4 · m4a · wav · ogg · mp3 · webm · aac</p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,video/*,.mp4,.m4a,.ogg,.webm,.wav,.aac,.mp3,.opus,.flac"
              multiple
              className="hidden"
              onChange={(e) => e.target.files && handleFiles(e.target.files)}
            />
          </div>
        )}

        {currentMuestras >= 3 && status !== 'recording' && status !== 'processing' && (
          <div className="mt-4 flex items-center justify-center gap-2 text-emerald-600 text-sm font-medium">
            <CheckCircle size={16} />
            <span>Frase lista</span>
          </div>
        )}
      </div>
    </div>
  )
}
