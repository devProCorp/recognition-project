export interface VocabularioEntry {
  significado: string
  urgencia: 'ALTA' | 'MEDIA' | 'BAJA'
  respuesta: string
}

export interface Vocabulario {
  [frase: string]: VocabularioEntry
}

export interface EstadoCalibracion {
  [frase: string]: number
}

export interface ResultadoAnalisis {
  frase_detectada: string
  significado: string
  urgencia: string
  respuesta: string
  confianza: 'ALTO' | 'MEDIO' | 'BAJO'
  distancia_dtw: number
  segunda_opcion: string | null
  dist_segunda: number | null
}

export interface EntradaHistorial {
  frase: string
  significado: string
  urgencia: string
}

export interface SegmentoDetectado {
  frase: string
  inicio: number
  fin: number
  ref_id: string | null
  fiore_id: string
  fiore_inicio: number
  fiore_fin: number
  duracion_fiore: number
  significado: string
  urgencia: string
}

export interface ResultadoSegmentacion {
  texto: string
  total_palabras: number
  frases_encontradas: number
  segmentos: SegmentoDetectado[]
}
