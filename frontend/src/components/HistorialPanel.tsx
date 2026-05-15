import { useEffect, useState } from 'react'
import { Trash2 } from 'lucide-react'
import { getHistorial, limpiarHistorial } from '../api'
import type { EntradaHistorial } from '../types'

const urgencyDot: Record<string, string> = {
  ALTA: 'bg-red-500',
  MEDIA: 'bg-amber-400',
  BAJA: 'bg-emerald-500',
}

export function HistorialPanel() {
  const [historial, setHistorial] = useState<EntradaHistorial[]>([])
  const [clearing, setClearing] = useState(false)

  const fetchHistorial = async () => {
    try {
      const data = await getHistorial()
      setHistorial(data.slice(-10).reverse())
    } catch {
    }
  }

  useEffect(() => {
    fetchHistorial()
    const interval = setInterval(fetchHistorial, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleLimpiar = async () => {
    setClearing(true)
    try {
      await limpiarHistorial()
      setHistorial([])
    } catch {
    } finally {
      setClearing(false)
    }
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
        <div>
          <h3 className="text-sm font-semibold text-slate-700">
            Historial de detecciones
          </h3>
          <p className="text-xs text-slate-400">Últimas 10 entradas</p>
        </div>
        <button
          onClick={handleLimpiar}
          disabled={clearing || historial.length === 0}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-red-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors px-2 py-1 rounded-lg hover:bg-red-50"
        >
          <Trash2 size={13} />
          Limpiar historial
        </button>
      </div>

      {historial.length === 0 ? (
        <div className="px-4 py-8 text-center text-slate-400 text-sm">
          Sin detecciones registradas
        </div>
      ) : (
        <ul className="divide-y divide-slate-50">
          {historial.map((entrada, idx) => (
            <li
              key={idx}
              className="flex items-start gap-3 px-4 py-3 hover:bg-slate-50 transition-colors"
            >
              <span
                className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${
                  urgencyDot[entrada.urgencia] ?? 'bg-slate-400'
                }`}
              />
              <div className="min-w-0">
                <p className="text-sm font-semibold text-slate-800 truncate">
                  {entrada.frase}
                </p>
                <p className="text-xs text-slate-500 truncate">
                  {entrada.significado}
                </p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
