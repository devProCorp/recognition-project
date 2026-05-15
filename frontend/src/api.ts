import type {
  Vocabulario,
  EstadoCalibracion,
  ResultadoAnalisis,
  EntradaHistorial,
  ResultadoSegmentacion,
} from './types'

export async function getVocabulario(): Promise<Vocabulario> {
  const res = await fetch('/vocabulario')
  if (!res.ok) throw new Error('Error al obtener vocabulario')
  return res.json()
}

export async function getEstadoCalibracion(): Promise<EstadoCalibracion> {
  const res = await fetch('/estado_calibracion')
  if (!res.ok) throw new Error('Error al obtener estado de calibración')
  return res.json()
}

export async function calibrar(
  frase: string,
  blob: Blob,
): Promise<{ ok: boolean; frase: string; muestras: number; listo: boolean }> {
  const formData = new FormData()
  formData.append('frase', frase)
  formData.append('audio', blob, 'audio.webm')
  const res = await fetch('/calibrar', { method: 'POST', body: formData })
  if (!res.ok) throw new Error('Error al calibrar')
  return res.json()
}

export async function analizar(blob: Blob): Promise<ResultadoAnalisis> {
  const formData = new FormData()
  formData.append('audio', blob, 'audio.webm')
  const res = await fetch('/analizar', { method: 'POST', body: formData })
  if (!res.ok) throw new Error('Error al analizar audio')
  return res.json()
}

export async function getHistorial(): Promise<EntradaHistorial[]> {
  const res = await fetch('/historial')
  if (!res.ok) throw new Error('Error al obtener historial')
  return res.json()
}

export async function limpiarHistorial(): Promise<{ ok: boolean }> {
  const res = await fetch('/historial', { method: 'DELETE' })
  if (!res.ok) throw new Error('Error al limpiar historial')
  return res.json()
}

export async function segmentar(file: File): Promise<ResultadoSegmentacion> {
  const formData = new FormData()
  formData.append('audio', file, file.name)
  const res = await fetch('/segmentar', { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as { error?: string }).error ?? 'Error al segmentar')
  }
  return res.json()
}

export async function calibrarSegmento(
  segId: string,
): Promise<{ ok: boolean; frase: string; muestras: number; listo: boolean }> {
  const res = await fetch(`/segmento/${segId}/calibrar`, { method: 'POST' })
  if (!res.ok) throw new Error('Error al agregar segmento')
  return res.json()
}

export function urlSegmento(segId: string): string {
  return `/segmento/${segId}`
}
