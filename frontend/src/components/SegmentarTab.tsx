import { useRef, useState } from 'react'
import {
  FileAudio,
  Loader2,
  Play,
  Pause,
  CheckCircle,
  AlertCircle,
  Scissors,
  ChevronDown,
  ChevronUp,
  Volume2,
  Mic,
  Settings2,
} from 'lucide-react'
import { segmentar, calibrarSegmento, urlSegmento } from '../api'
import type { ResultadoSegmentacion, SegmentoDetectado } from '../types'

type SegmentoState = 'idle' | 'adding' | 'added' | 'error'

const urgencyBadge: Record<string, string> = {
  ALTA: 'bg-red-100 text-red-700',
  MEDIA: 'bg-amber-100 text-amber-700',
  BAJA: 'bg-blue-100 text-blue-700',
}

function fmt(s: number) {
  const m = Math.floor(s / 60)
  const sec = (s % 60).toFixed(1).padStart(4, '0')
  return `${m}:${sec}`
}

function MiniPlayer({
  segId,
  label,
  icon,
  color,
}: {
  segId: string
  label: string
  icon: React.ReactNode
  color: string
}) {
  const ref = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)

  const toggle = () => {
    if (!ref.current) return
    if (playing) { ref.current.pause(); setPlaying(false) }
    else { ref.current.play(); setPlaying(true) }
  }

  return (
    <div className={`flex items-center gap-2 rounded-lg px-3 py-2 ${color}`}>
      <div className="flex items-center gap-1.5 text-xs font-semibold opacity-70">
        {icon}
        <span>{label}</span>
      </div>
      <button
        onClick={toggle}
        className="ml-auto w-7 h-7 rounded-full bg-white/60 hover:bg-white flex items-center justify-center transition-colors flex-shrink-0"
      >
        {playing
          ? <Pause size={13} className="text-slate-700" />
          : <Play size={13} className="text-slate-700 ml-0.5" />}
      </button>
      <audio ref={ref} src={urlSegmento(segId)} onEnded={() => setPlaying(false)} preload="none" />
    </div>
  )
}

function SegmentoCard({
  seg,
  onAdded,
}: {
  seg: SegmentoDetectado
  onAdded: (frase: string, muestras: number) => void
}) {
  const [state, setState] = useState<SegmentoState>('idle')
  const [muestras, setMuestras] = useState<number | null>(null)

  const handleAgregar = async () => {
    setState('adding')
    try {
      const res = await calibrarSegmento(seg.fiore_id)
      setMuestras(res.muestras)
      onAdded(seg.frase, res.muestras)
      setState('added')
    } catch {
      setState('error')
    }
  }

  return (
    <div className={`rounded-xl border bg-white p-3 shadow-sm flex flex-col gap-2.5 transition-all ${state === 'added' ? 'ring-2 ring-emerald-400 ring-offset-1' : 'border-slate-200'}`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="font-semibold text-slate-800 text-sm leading-tight">{seg.frase}</p>
          <p className="text-xs text-slate-400 mt-0.5">{seg.significado}</p>
        </div>
        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full flex-shrink-0 ${urgencyBadge[seg.urgencia] ?? urgencyBadge.BAJA}`}>
          {seg.urgencia}
        </span>
      </div>

      {/* Timestamps */}
      <div className="flex items-center gap-1 text-[11px] text-slate-400 font-mono">
        <span>ref {fmt(seg.inicio)}–{fmt(seg.fin)}</span>
        <span className="mx-1 text-slate-200">|</span>
        <span>fiore {fmt(seg.fiore_inicio)}–{fmt(seg.fiore_fin)}</span>
        <span className="ml-auto">{seg.duracion_fiore.toFixed(1)}s</span>
      </div>

      {/* Players */}
      <div className="flex flex-col gap-1.5">
        {seg.ref_id && (
          <MiniPlayer
            segId={seg.ref_id}
            label="Voz de referencia"
            icon={<Volume2 size={11} />}
            color="bg-slate-100 text-slate-600"
          />
        )}
        <MiniPlayer
          segId={seg.fiore_id}
          label="Respuesta de Fiore"
          icon={<Mic size={11} />}
          color="bg-violet-50 text-violet-700"
        />
      </div>

      {/* Action */}
      <button
        onClick={state === 'error' ? handleAgregar : state === 'idle' ? handleAgregar : undefined}
        disabled={state === 'adding' || state === 'added'}
        className={`w-full py-1.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors ${
          state === 'added'
            ? 'bg-emerald-100 text-emerald-700 cursor-default'
            : state === 'error'
              ? 'bg-red-100 text-red-600 cursor-pointer'
              : state === 'adding'
                ? 'bg-violet-100 text-violet-500 cursor-not-allowed'
                : 'bg-violet-600 text-white hover:bg-violet-700 cursor-pointer'
        }`}
      >
        {state === 'adding' && <Loader2 size={12} className="animate-spin" />}
        {state === 'added' && <CheckCircle size={12} />}
        {state === 'error' && <AlertCircle size={12} />}
        {state === 'idle' && 'Agregar respuesta de Fiore a calibración'}
        {state === 'adding' && 'Agregando…'}
        {state === 'added' && `Agregado (${muestras} muestra${muestras !== 1 ? 's' : ''})`}
        {state === 'error' && 'Error — reintentar'}
      </button>
    </div>
  )
}

export function SegmentarTab() {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [resultado, setResultado] = useState<ResultadoSegmentacion | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [frasesAgregadas, setFrasesAgregadas] = useState<Record<string, number>>({})
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [showConfig, setShowConfig] = useState(false)
  const [ventana, setVentana] = useState(4.5)
  const [gap, setGap] = useState(0.3)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setResultado(null)
    setError(null)
    setFrasesAgregadas({})
    setCollapsed({})
  }

  const procesar = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResultado(null)
    try {
      setResultado(await segmentar(file, ventana, gap))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  const handleAdded = (frase: string, muestras: number) => {
    setFrasesAgregadas((prev) => ({ ...prev, [frase]: muestras }))
  }

  const toggleCollapse = (frase: string) =>
    setCollapsed((prev) => ({ ...prev, [frase]: !prev[frase] }))

  const porFrase: Record<string, SegmentoDetectado[]> = {}
  if (resultado) {
    for (const seg of resultado.segmentos) {
      if (!porFrase[seg.frase]) porFrase[seg.frase] = []
      porFrase[seg.frase].push(seg)
    }
  }

  const totalAgregados = Object.keys(frasesAgregadas).length

  return (
    <div className="space-y-5">
      <div>
        <h2 className="font-bold text-slate-800 text-base">Segmentador automático</h2>
        <p className="text-sm text-slate-500 mt-0.5">
          Sube el audio donde una voz normal dice cada frase y Fiore la repite. Whisper detecta la
          voz normal, y el sistema extrae automáticamente la respuesta de Fiore para calibración.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          const f = e.dataTransfer.files[0]
          if (f) handleFile(f)
        }}
        onClick={() => !loading && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors ${
          loading ? 'opacity-60 cursor-not-allowed border-slate-300'
          : isDragging ? 'border-violet-400 bg-violet-50'
          : file ? 'border-emerald-400 bg-emerald-50'
          : 'border-slate-300 hover:border-violet-400 hover:bg-slate-50'
        }`}
      >
        <FileAudio size={36} className={isDragging ? 'text-violet-500' : file ? 'text-emerald-500' : 'text-slate-400'} />
        {file ? (
          <>
            <p className="font-semibold text-slate-700 text-sm text-center">{file.name}</p>
            <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          </>
        ) : (
          <>
            <p className="font-semibold text-slate-600 text-sm">Arrastra el audio aquí</p>
            <p className="text-xs text-slate-400">mp4 · m4a · wav · ogg · mp3</p>
          </>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,video/*,.mp4,.m4a,.ogg,.webm,.wav,.aac,.mp3"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
      </div>

      {file && !loading && (
        <div className="flex gap-2">
          <button
            onClick={procesar}
            className="flex-1 py-3 bg-violet-600 hover:bg-violet-700 text-white font-semibold rounded-xl flex items-center justify-center gap-2 transition-colors"
          >
            <Scissors size={16} />
            Detectar frases y extraer respuestas de Fiore
          </button>
          <button
            onClick={() => setShowConfig((v) => !v)}
            className={`px-3 py-3 rounded-xl border transition-colors ${showConfig ? 'border-violet-400 bg-violet-50 text-violet-700' : 'border-slate-200 text-slate-500 hover:border-slate-300'}`}
            title="Ajustes"
          >
            <Settings2 size={18} />
          </button>
        </div>
      )}

      {/* Config panel */}
      {showConfig && (
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-3">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Ajuste de tiempos</p>
          <div className="grid grid-cols-2 gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 font-medium">Ventana Fiore (s)</span>
              <span className="text-xs text-slate-400">Duración máxima esperada de la respuesta</span>
              <input
                type="number" min={1} max={10} step={0.5}
                value={ventana}
                onChange={(e) => setVentana(Number(e.target.value))}
                className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-violet-400"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 font-medium">Gap inicial (s)</span>
              <span className="text-xs text-slate-400">Pausa mínima tras la voz normal</span>
              <input
                type="number" min={0} max={3} step={0.1}
                value={gap}
                onChange={(e) => setGap(Number(e.target.value))}
                className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-violet-400"
              />
            </label>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center gap-3 py-10 text-slate-500">
          <Loader2 size={32} className="animate-spin text-violet-500" />
          <p className="text-sm font-medium">Transcribiendo con Whisper…</p>
          <p className="text-xs text-slate-400">Puede tardar hasta 1 minuto por cada 5 min de audio</p>
        </div>
      )}

      {error && (
        <div className="rounded-xl bg-red-50 border border-red-200 p-4 flex gap-3 items-start">
          <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-red-700 text-sm">Error al procesar</p>
            <p className="text-xs text-red-500 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {resultado && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="rounded-xl bg-slate-50 border border-slate-200 p-4 flex flex-wrap gap-4 text-sm">
            <div className="flex flex-col">
              <span className="text-xs text-slate-400 uppercase tracking-wide font-semibold">Palabras</span>
              <span className="font-bold text-slate-800 text-lg">{resultado.total_palabras}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-slate-400 uppercase tracking-wide font-semibold">Frases detectadas</span>
              <span className="font-bold text-violet-700 text-lg">{resultado.frases_encontradas}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-slate-400 uppercase tracking-wide font-semibold">Segmentos</span>
              <span className="font-bold text-slate-800 text-lg">{resultado.segmentos.length}</span>
            </div>
            {totalAgregados > 0 && (
              <div className="flex flex-col">
                <span className="text-xs text-slate-400 uppercase tracking-wide font-semibold">Calibradas</span>
                <span className="font-bold text-emerald-600 text-lg">{totalAgregados}</span>
              </div>
            )}
          </div>

          {resultado.texto && (
            <details className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              <summary className="px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide cursor-pointer hover:bg-slate-50">
                Transcripción Whisper
              </summary>
              <p className="px-4 pb-3 text-sm text-slate-600 leading-relaxed">{resultado.texto}</p>
            </details>
          )}

          {resultado.segmentos.length === 0 && (
            <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 text-sm text-amber-700">
              No se encontraron frases del vocabulario. Intenta ajustar la ventana o verifica que el audio contenga las frases de referencia dichas con claridad.
            </div>
          )}

          {Object.entries(porFrase).map(([frase, segs]) => (
            <div key={frase} className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              <button
                onClick={() => toggleCollapse(frase)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors"
              >
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-semibold text-slate-800 text-sm">{frase}</span>
                  <span className="text-xs text-slate-400">{segs.length} ocurrencia{segs.length !== 1 ? 's' : ''}</span>
                  {frasesAgregadas[frase] !== undefined && (
                    <span className="flex items-center gap-1 text-xs text-emerald-600 font-semibold">
                      <CheckCircle size={12} />
                      {frasesAgregadas[frase]} muestra{frasesAgregadas[frase] !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>
                {collapsed[frase]
                  ? <ChevronDown size={16} className="text-slate-400 flex-shrink-0" />
                  : <ChevronUp size={16} className="text-slate-400 flex-shrink-0" />}
              </button>

              {!collapsed[frase] && (
                <div className="px-4 pb-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {segs.map((seg, idx) => (
                    <SegmentoCard key={`${seg.fiore_id}-${idx}`} seg={seg} onAdded={handleAdded} />
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
