import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        border: "var(--border)",
        divider: "var(--divider)",
        text: "var(--text)",
        "text-2": "var(--text-2)",
        "text-3": "var(--text-3)",
        accent: "var(--accent)",
        "accent-2": "var(--accent-2)",
        consensus: "var(--consensus)",
        ok: "var(--ok)",
        warn: "var(--warn)",
        alarm: "var(--alarm)",
        info: "var(--info)",
      },
      fontFamily: {
        display: ["Inter Tight", "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
        num: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      fontVariantNumeric: {
        tabular: "tabular-nums",
      },
      borderRadius: {
        sm: "4px",
        md: "6px",
        lg: "10px",
        xl: "16px",
      },
      boxShadow: {
        1: "0 1px 2px rgb(0 0 0 / 0.04), 0 1px 1px rgb(0 0 0 / 0.03)",
        2: "0 2px 6px rgb(0 0 0 / 0.06), 0 1px 2px rgb(0 0 0 / 0.04)",
        3: "0 8px 24px rgb(0 0 0 / 0.08), 0 2px 8px rgb(0 0 0 / 0.05)",
      },
      transitionTimingFunction: {
        clinical: "cubic-bezier(0.22, 0.61, 0.36, 1)",
      },
    },
  },
  plugins: [],
} satisfies Config;
