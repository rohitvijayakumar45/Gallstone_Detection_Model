import clsx from "clsx";
import type { Layer } from "@/lib/types";

interface Props {
  value: Record<Layer, boolean>;
  onChange: (next: Record<Layer, boolean>) => void;
  heatmapAvailable: boolean;
}

const items: { key: Layer; label: string; hotkey: string }[] = [
  { key: "raw", label: "Raw", hotkey: "0" },
  { key: "yolo", label: "YOLO", hotkey: "Y" },
  { key: "rfdetr", label: "RF-DETR", hotkey: "R" },
  { key: "consensus", label: "Consensus", hotkey: "C" },
  { key: "conformal", label: "Conformal", hotkey: "X" },
  { key: "heatmap", label: "Heatmap", hotkey: "H" },
];

export default function LayerToggles({ value, onChange, heatmapAvailable }: Props) {
  return (
    <div className="flex items-center gap-1">
      {items.map((it) => {
        const disabled = it.key === "heatmap" && !heatmapAvailable;
        return (
          <button
            key={it.key}
            type="button"
            disabled={disabled}
            onClick={() => onChange({ ...value, [it.key]: !value[it.key] })}
            className={clsx(
              "h-7 px-2.5 rounded-sm text-[11px] transition-colors duration-150 ease-clinical border",
              value[it.key]
                ? "bg-accent text-white border-accent"
                : "bg-surface text-text-2 border-border hover:text-text hover:bg-surface-2",
              disabled && "opacity-40 cursor-not-allowed",
            )}
            title={`Toggle ${it.label} (${it.hotkey})`}
          >
            <span className="label-caps mr-1" style={{ color: "currentColor" }}>
              {it.hotkey}
            </span>
            {it.label}
          </button>
        );
      })}
    </div>
  );
}
