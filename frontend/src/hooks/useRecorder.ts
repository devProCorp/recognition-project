import { useState, useRef, useCallback } from 'react'

interface UseRecorderReturn {
  isRecording: boolean
  audioBlob: Blob | null
  error: string | null
  startRecording: () => Promise<void>
  stopRecording: () => void
  reset: () => void
}

export function useRecorder(): UseRecorderReturn {
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== 'inactive'
    ) {
      mediaRecorderRef.current.stop()
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    setIsRecording(false)
  }, [])

  const startRecording = useCallback(async () => {
    setError(null)
    setAudioBlob(null)
    chunksRef.current = []

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('No se pudo acceder al micrófono.')
      return
    }

    streamRef.current = stream

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : ''

    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {})
    mediaRecorderRef.current = recorder

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data)
      }
    }

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, {
        type: mimeType || 'audio/webm',
      })
      setAudioBlob(blob)
      setIsRecording(false)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
    }

    recorder.start()
    setIsRecording(true)

    timerRef.current = setTimeout(() => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== 'inactive'
      ) {
        mediaRecorderRef.current.stop()
      }
      timerRef.current = null
    }, 4000)
  }, [])

  const reset = useCallback(() => {
    setAudioBlob(null)
    setError(null)
    setIsRecording(false)
  }, [])

  return { isRecording, audioBlob, error, startRecording, stopRecording, reset }
}
