import { useEffect, useState, useCallback } from 'react'
import { Mic2 } from 'lucide-react'
import { getVocabulario, getEstadoCalibracion } from './api'
import type { Vocabulario, EstadoCalibracion } from './types'
import { CalibrarTab } from './components/CalibrarTab'
import { AnalizarTab } from './components/AnalizarTab'
import { HistorialPanel } from './components/HistorialPanel'
import { SegmentarTab } from './components/SegmentarTab'

type Tab = 'analizar' | 'calibrar' | 'segmentar'

export default function App() {
  const [tab, setTab] = useState<Tab>('analizar')
  const [vocabulario, setVocabulario] = useState<Vocabulario>({})
  const [estadoCalibracion, setEstadoCalibracion] = useState<EstadoCalibracion>({})
  const [loadingVocab, setLoadingVocab] = useState(true)

  const totalFrases = Object.keys(vocabulario).length
  const frasesListas = Object.values(estadoCalibracion).filter((n) => n >= 3).length

  const fetchEstado = useCallback(async () => {
    try {
      const estado = await getEstadoCalibracion()
      setEstadoCalibracion(estado)
    } catch {
    }
  }, [])

  useEffect(() => {
    Promise.all([getVocabulario(), getEstadoCalibracion()])
      .then(([vocab, estado]) => {
        setVocabulario(vocab)
        setEstadoCalibracion(estado)
      })
      .catch(() => {})
      .finally(() => setLoadingVocab(false))
  }, [])

  useEffect(() => {
    const interval = setInterval(fetchEstado, 5000)
    return () => clearInterval(interval)
  }, [fetchEstado])

  const handleEstadoUpdate = (frase: string, count: number) => {
    setEstadoCalibracion((prev) => ({ ...prev, [frase]: count }))
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-gradient-to-r from-violet-700 to-purple-800 shadow-lg">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center">
              <Mic2 size={22} className="text-white" />
            </div>
            <div>
              <h1 className="text-white font-bold text-lg leading-tight">
                Analista de Voz · Fiore
              </h1>
              <p className="text-violet-200 text-xs">
                Sistema de reconocimiento de voz asistivo
              </p>
            </div>
          </div>

          {!loadingVocab && totalFrases > 0 && (
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold ${
                frasesListas >= totalFrases
                  ? 'bg-emerald-500 text-white'
                  : frasesListas >= totalFrases / 2
                    ? 'bg-amber-400 text-white'
                    : 'bg-white/20 text-white'
              }`}
            >
              <span>
                {frasesListas}/{totalFrases} calibradas
              </span>
            </div>
          )}
        </div>

        <div className="max-w-5xl mx-auto px-4">
          <div className="flex gap-1">
            {([
              ['analizar', 'Analizar'],
              ['calibrar', 'Calibrar'],
              ['segmentar', 'Segmentar audio'],
            ] as [Tab, string][]).map(([t, label]) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-5 py-2.5 text-sm font-semibold transition-all focus:outline-none ${
                  tab === t
                    ? 'text-white border-b-2 border-white'
                    : 'text-violet-200 hover:text-white border-b-2 border-transparent'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-6">
        {loadingVocab ? (
          <div className="flex items-center justify-center py-20 text-slate-400">
            Cargando vocabulario...
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              {tab === 'analizar' && <AnalizarTab />}
              {tab === 'calibrar' && (
                <CalibrarTab
                  vocabulario={vocabulario}
                  estadoCalibracion={estadoCalibracion}
                  onEstadoUpdate={handleEstadoUpdate}
                />
              )}
              {tab === 'segmentar' && <SegmentarTab />}
            </div>
            <div className="lg:col-span-1">
              <HistorialPanel />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
