import { NextResponse } from 'next/server'

let filaData = { pessoas: 0, tempo_medio_espera: 0, timestamp: null }

export async function GET() {
  return NextResponse.json(filaData)
}

export async function POST(request: Request) {
  const data = await request.json()
  filaData = data
  return NextResponse.json({ status: 'ok' })
}
