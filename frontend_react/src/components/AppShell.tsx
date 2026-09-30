import { NavLink } from "react-router-dom";
import type { ReactNode } from "react";
import clsx from "clsx";

const nav = [
  { to: "/", label: "Overview", end: true },
  { to: "/study", label: "Workstation" },
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/audit", label: "Audit" },
];

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-bg text-text">
      <header className="h-14 border-b border-border bg-surface/95 backdrop-blur px-6 flex items-center gap-8">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ background: "var(--accent)" }}
          />
          <span className="font-display font-semibold tracking-tight text-[15px]">
            GallStone Workstation
          </span>
          <span className="ml-2 label-caps">v6 · patent-grade</span>
        </div>
        <nav className="flex items-center gap-1">
          {nav.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                clsx(
                  "px-3 h-8 rounded-md inline-flex items-center text-[13px] transition-colors duration-150 ease-clinical",
                  isActive
                    ? "bg-surface-2 text-text"
                    : "text-text-2 hover:text-text hover:bg-surface-2",
                )
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-[12px] text-text-3">
          <span className="num">API · localhost:8000</span>
          <span
            aria-hidden
            className="inline-block w-2 h-2 rounded-full"
            style={{ background: "var(--ok)" }}
          />
        </div>
      </header>
      <main className="flex-1 min-h-0">{children}</main>
      <footer className="h-10 border-t border-border px-6 flex items-center text-[12px] text-text-3">
        <span>Decision-support only. Clinician review required for all findings.</span>
        <span className="ml-auto num">
          model v6.0 · calibration v1 · conformal α=0.05
        </span>
      </footer>
    </div>
  );
}
