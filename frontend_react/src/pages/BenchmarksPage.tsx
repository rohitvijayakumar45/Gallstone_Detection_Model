import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import ReliabilityDiagram from "@/components/ReliabilityDiagram";

interface CalReport {
  yolo?: { T: number; before: any; after: any };
  rfdetr?: { T: number; before: any; after: any };
  fusion?: { before: any; after: any };
}

interface ConformalReport {
  alpha: number;
  split_conformal: {
    q: number[];
    n_calibration: number;
    bonferroni: boolean;
    empirical_coverage_test: number;
    target_coverage: number;
    n_test_pairs: number;
  };
  crc: {
    lambda_star: number;
    n_calibration_images: number;
    empirical_fnr_test: number;
    target_fnr: number;
  };
}

interface BaselineReport {
  split: string;
  data_dir: string;
  metric: string;
  baseline: number;
  results: Record<string, number>;
  final: number;
  improvement: number;
}

function fmt(v: number | null | undefined, d = 4): string {
  return v == null || isNaN(v) ? "—" : v.toFixed(d);
}

export default function BenchmarksPage() {
  const baseline = useQuery({
    queryKey: ["metrics", "benchmarks"],
    queryFn: () => api.benchmarks() as Promise<BaselineReport>,
    retry: false,
  });
  const calibration = useQuery({
    queryKey: ["metrics", "calibration"],
    queryFn: () => api.calibration() as Promise<CalReport>,
    retry: false,
  });
  const conformal = useQuery({
    queryKey: ["metrics", "conformal"],
    queryFn: () => api.conformal() as Promise<ConformalReport>,
    retry: false,
  });

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10">
      <header>
        <p className="label-caps mb-2">Benchmarks</p>
        <h2 className="font-display text-2xl mb-2">Regenerated from repo</h2>
        <p className="text-[13px] text-text-2 max-w-[70ch]">
          Every number below comes from a JSON under{" "}
          <code className="num">runs/final_eval/</code>. Blanks mean the
          matching fit script has not been run yet — see{" "}
          <code className="num">python scripts/regenerate_all_numbers.py</code>.
        </p>
      </header>

      <section>
        <p className="label-caps mb-3">Ablation ladder</p>
        {baseline.isError ? (
          <p className="text-[12px] text-text-3">
            No baseline report yet. Run{" "}
            <code className="num">scripts/comprehensive_evaluation.py</code>.
          </p>
        ) : baseline.isLoading ? (
          <p className="text-[12px] text-text-3">Loading…</p>
        ) : (
          <div>
            <p className="text-[12px] text-text-3 mb-2 num">
              split={baseline.data?.split} · metric={baseline.data?.metric}
            </p>
            <table className="w-full border border-border rounded-md overflow-hidden">
              <thead className="bg-surface-2">
                <tr className="text-left">
                  <th className="px-4 py-2 text-[12px] font-medium">Config</th>
                  <th className="px-4 py-2 text-[12px] font-medium text-right">Metric</th>
                  <th className="px-4 py-2 text-[12px] font-medium text-right">Δ vs baseline</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(baseline.data?.results ?? {}).map(([k, v]) => (
                  <tr key={k} className="border-t border-divider bg-surface">
                    <td className="px-4 py-2 text-[13px]">{k}</td>
                    <td className="px-4 py-2 text-right num">{fmt(v)}</td>
                    <td className="px-4 py-2 text-right num text-text-3">
                      {fmt(v - (baseline.data?.baseline ?? 0))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <p className="label-caps mb-3">Calibration · val=368</p>
        {calibration.isError ? (
          <p className="text-[12px] text-text-3">
            No calibration report. Run{" "}
            <code className="num">scripts/fit_calibration.py</code>.
          </p>
        ) : calibration.isLoading ? (
          <p className="text-[12px] text-text-3">Loading…</p>
        ) : (
          <div className="space-y-6">
            <table className="w-full border border-border rounded-md overflow-hidden">
              <thead className="bg-surface-2">
                <tr className="text-left">
                  <th className="px-4 py-2 text-[12px]">Model</th>
                  <th className="px-4 py-2 text-[12px] text-right">T</th>
                  <th className="px-4 py-2 text-[12px] text-right">ECE before</th>
                  <th className="px-4 py-2 text-[12px] text-right">ECE after</th>
                  <th className="px-4 py-2 text-[12px] text-right">Brier after</th>
                </tr>
              </thead>
              <tbody>
                {(["yolo", "rfdetr", "fusion"] as const).map((k) => {
                  const c = (calibration.data as any)?.[k];
                  const before = c?.before;
                  const after = c?.after;
                  return (
                    <tr key={k} className="border-t border-divider bg-surface">
                      <td className="px-4 py-2 text-[13px]">{k}</td>
                      <td className="px-4 py-2 text-right num">{fmt(c?.T, 3) ?? "—"}</td>
                      <td className="px-4 py-2 text-right num">{fmt(before?.ece)}</td>
                      <td className="px-4 py-2 text-right num text-ok">{fmt(after?.ece)}</td>
                      <td className="px-4 py-2 text-right num">{fmt(after?.brier)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(["yolo", "rfdetr", "fusion"] as const).flatMap((k) => {
                const c = (calibration.data as any)?.[k];
                const beforeBins = c?.before?.reliability;
                const afterBins = c?.after?.reliability;
                if (!beforeBins || !afterBins) return [];
                return [
                  <ReliabilityDiagram key={`${k}-b`} bins={beforeBins} title={`${k} · before`} />,
                  <ReliabilityDiagram key={`${k}-a`} bins={afterBins} title={`${k} · after`} />,
                ];
              })}
            </div>
          </div>
        )}
      </section>

      <section>
        <p className="label-caps mb-3">Conformal · α=0.05</p>
        {conformal.isError ? (
          <p className="text-[12px] text-text-3">
            No conformal report. Run{" "}
            <code className="num">scripts/fit_conformal.py</code>.
          </p>
        ) : conformal.isLoading ? (
          <p className="text-[12px] text-text-3">Loading…</p>
        ) : conformal.data ? (
          <div className="grid grid-cols-2 gap-4">
            <div className="card p-4">
              <p className="label-caps mb-2">Split-conformal box</p>
              <dl className="grid grid-cols-2 gap-y-1 text-[13px]">
                <dt className="text-text-3">q_x1</dt>
                <dd className="text-right num">{fmt(conformal.data.split_conformal.q[0])}</dd>
                <dt className="text-text-3">q_y1</dt>
                <dd className="text-right num">{fmt(conformal.data.split_conformal.q[1])}</dd>
                <dt className="text-text-3">q_x2</dt>
                <dd className="text-right num">{fmt(conformal.data.split_conformal.q[2])}</dd>
                <dt className="text-text-3">q_y2</dt>
                <dd className="text-right num">{fmt(conformal.data.split_conformal.q[3])}</dd>
                <dt className="text-text-3">Target coverage</dt>
                <dd className="text-right num">{fmt(conformal.data.split_conformal.target_coverage, 3)}</dd>
                <dt className="text-text-3">Empirical (test)</dt>
                <dd className="text-right num text-ok">{fmt(conformal.data.split_conformal.empirical_coverage_test, 3)}</dd>
              </dl>
            </div>
            <div className="card p-4">
              <p className="label-caps mb-2">CRC recall bound</p>
              <dl className="grid grid-cols-2 gap-y-1 text-[13px]">
                <dt className="text-text-3">λ*</dt>
                <dd className="text-right num">{fmt(conformal.data.crc.lambda_star, 3)}</dd>
                <dt className="text-text-3">Target FNR</dt>
                <dd className="text-right num">{fmt(conformal.data.crc.target_fnr, 3)}</dd>
                <dt className="text-text-3">Empirical FNR (test)</dt>
                <dd className="text-right num text-ok">{fmt(conformal.data.crc.empirical_fnr_test, 3)}</dd>
                <dt className="text-text-3">n calibration</dt>
                <dd className="text-right num">{conformal.data.crc.n_calibration_images}</dd>
              </dl>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}
