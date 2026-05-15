import { useState } from 'react'
import { CheckCircle } from 'lucide-react'
import type { Vocabulario, EstadoCalibracion } from '../types'
import { RecorderModal } from './RecorderModal'

interface Props {
  vocabulario: Vocabulario
  estadoCalibracion: EstadoCalibracion
  onEstadoUpdate: (frase: string, count: number) => void
}

const URGENCY_ORDER = ['ALTA', 'MEDIA', 'BAJA'] as const

const urgencyLabel: Record<string, string> = {
  ALTA: 'Urgencia Alta',
  MEDIA: 'Urgencia Media',
  BAJA: 'Urgencia Baja',
}

const urgencyHeaderClass: Record<string, string> = {
  ALTA: 'text-red-600 border-red-200 bg-red-50',
  MEDIA: 'text-amber-600 border-amber-200 bg-amber-50',
  BAJA: 'text-blue-600 border-blue-200 bg-blue-50',
}

const urgencyBadgeClass: Record<string, string> = {
  ALTA: 'bg-red-100 text-red-700',
  MEDIA: 'bg-amber-100 text-amber-700',
  BAJA: 'bg-blue-100 text-blue-700',
}

export function CalibrarTab({ vocabulario, estadoCalibracion, onEstadoUpdate }: Props) {
  const [modalFrase, setModalFrase] = useState<string | null>(null)

  const byUrgency = URGENCY_ORDER.reduce<Record<string, string[]>>(
    (acc, u) => {
      acc[u] = Object.entries(vocabulario)
        .filter(([, entry]) => entry.urgencia === u)
        .map(([frase]) => frase)
      return acc
    },
    {},
  )

  const modalEntry = modalFrase
    ? {
        frase: modalFrase,
        muestras: estadoCalibracion[modalFrase] ?? 0,
      }
    : null

  return (
    <div className="space-y-8">
      {URGENCY_ORDER.map((urgencia) => {
        const frases = byUrgency[urgencia]
        if (!frases.length) return null
        return (
          <section key={urgencia}>
            <div
              className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-semibold uppercase tracking-wide mb-4 ${urgencyHeaderClass[urgencia]}`}
            >
              {urgencyLabel[urgencia]}
              <span className="opacity-60">({frases.length})</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {frases.map((frase) => {
                const count = estadoCalibracion[frase] ?? 0
                const listo = count >= 3
                return (
                  <button
                    key={frase}
                    onClick={() => setModalFrase(frase)}
                    className={`text-left p-3 rounded-xl border bg-white shadow-sm hover:shadow-md transition-all focus:outline-none focus:ring-2 focus:ring-violet-400 ${
                      listo
                        ? 'ring-2 ring-emerald-400 ring-offset-1'
                        : 'border-slate-200'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-1 mb-2">
                      <span className="font-semibold text-sm text-slate-800 leading-tight">
                        {frase}
                      </span>
                      {listo && (
                        <CheckCircle
                          size={14}
                          className="text-emerald-500 flex-shrink-0 mt-0.5"
                        />
                      )}
                    </div>
                    <span
                      className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${urgencyBadgeClass[urgencia]}`}
                    >
                      {urgencia}
                    </span>
                    <div className="flex gap-1.5 mt-3">
                      {[0, 1, 2].map((i) => (
                        <div
                          key={i}
                          className={`w-3 h-3 rounded-full transition-colors ${
                            i < count ? 'bg-emerald-500' : 'bg-slate-200'
                          }`}
                        />
                      ))}
                    </div>
                  </button>
                )
              })}
            </div>
          </section>
        )
      })}

      {modalEntry && (
        <RecorderModal
          frase={modalEntry.frase}
          muestras={modalEntry.muestras}
          onClose={() => setModalFrase(null)}
          onSampleAdded={(newCount) => {
            onEstadoUpdate(modalEntry.frase, newCount)
          }}
        />
      )}
    </div>
  )
}
