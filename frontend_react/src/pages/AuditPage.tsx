export default function AuditPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <p className="label-caps mb-2">Audit log</p>
      <h2 className="font-display text-2xl mb-4">Every scan, every provenance</h2>
      <p className="text-[13px] text-text-2 max-w-[60ch]">
        Each row records scan id, model SHA, calibration version, conformal α,
        clinician verdict, and timestamp. Immutable append-only backing store.
      </p>
      <div className="mt-8 card p-6 text-[12px] text-text-3 num">no rows yet</div>
    </div>
  );
}
