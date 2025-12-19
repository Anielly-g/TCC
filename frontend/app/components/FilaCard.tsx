"use client";
import useSWR from "swr";

const fetcher = (url: string) => fetch(url).then(res => res.json());

export default function FilaCard() {
  const { data, error } = useSWR("/api/fila", fetcher, { refreshInterval: 3000 });

  if (error) return <p>Erro ao carregar dados.</p>;
  if (!data) return <p>Carregando...</p>;

  const tempoMin = (data.tempo_medio_espera / 60).toFixed(1);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-100 p-6">
      <div className="bg-white shadow-xl rounded-2xl p-8 text-center w-full max-w-sm">
        <h1 className="text-2xl font-bold mb-4">📊 Monitor de Fila</h1>
        <p className="text-lg">Pessoas na fila: <b>{data.pessoas}</b></p>
        <p className="text-lg">Tempo médio: <b>{tempoMin} min</b></p>
        <p className="text-sm text-gray-500 mt-4">
          Última atualização:<br />
          {data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : "-"}
        </p>
      </div>
    </div>
  );
}
