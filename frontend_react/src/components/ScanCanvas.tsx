import { useEffect, useRef, useState } from "react";
import { Layer, Rect, Stage, Text, Image as KImage, Group } from "react-konva";
import Konva from "konva";
import type { Detection, Layer as LayerName } from "@/lib/types";

interface Props {
  previewUrl: string;
  detections: Detection[];
  layers: Record<LayerName, boolean>;
  heatmapUrl?: string | null;
}

const ACCENT = "#1F5C4C";
const BRONZE = "#7A5A2C";
const CONSENSUS = "#0F3D33";
const CONFORMAL = "#B7791F";

function useImage(src: string): [HTMLImageElement | null, { w: number; h: number }] {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [dims, setDims] = useState({ w: 0, h: 0 });
  useEffect(() => {
    if (!src) return;
    const el = new window.Image();
    el.crossOrigin = "anonymous";
    el.src = src;
    el.onload = () => {
      setImg(el);
      setDims({ w: el.naturalWidth, h: el.naturalHeight });
    };
  }, [src]);
  return [img, dims];
}

export default function ScanCanvas({ previewUrl, detections, layers, heatmapUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [img, dims] = useImage(previewUrl);
  const [heat] = useImage(heatmapUrl ?? "");
  const [stageSize, setStageSize] = useState({ w: 800, h: 600 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setStageSize({ w: r.width, h: r.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const scale =
    dims.w && dims.h
      ? Math.min(stageSize.w / dims.w, stageSize.h / dims.h)
      : 1;
  const offX = (stageSize.w - dims.w * scale) / 2;
  const offY = (stageSize.h - dims.h * scale) / 2;

  return (
    <div ref={containerRef} className="w-full h-full bg-surface-2/40">
      <Stage width={stageSize.w} height={stageSize.h}>
        <Layer>
          {img && layers.raw ? (
            <KImage image={img} x={offX} y={offY} width={dims.w * scale} height={dims.h * scale} />
          ) : null}
          {heat && layers.heatmap ? (
            <KImage
              image={heat}
              x={offX}
              y={offY}
              width={dims.w * scale}
              height={dims.h * scale}
              opacity={0.7}
              globalCompositeOperation="screen"
            />
          ) : null}
        </Layer>
        <Layer listening={false}>
          {detections.map((d, i) => {
            const b = d.bbox;
            const x = offX + b.x1 * scale;
            const y = offY + b.y1 * scale;
            const w = (b.x2 - b.x1) * scale;
            const h = (b.y2 - b.y1) * scale;
            const conf = d.calibrated_confidence ?? d.confidence;
            const color = layers.consensus ? CONSENSUS : ACCENT;
            const boxes: JSX.Element[] = [];
            if (layers.consensus || layers.yolo || layers.rfdetr) {
              boxes.push(
                <Rect key={`box-${i}`} x={x} y={y} width={w} height={h} stroke={color} strokeWidth={1.5} />,
              );
              boxes.push(
                <Text
                  key={`t-${i}`}
                  x={x + 4}
                  y={y - 16}
                  text={`${(conf * 100).toFixed(0)}%`}
                  fontFamily="JetBrains Mono, ui-monospace"
                  fontSize={11}
                  fill={color}
                />,
              );
            }
            if (layers.conformal && d.conformal_bbox) {
              const cb = d.conformal_bbox;
              boxes.push(
                <Rect
                  key={`c-${i}`}
                  x={offX + cb.x1 * scale}
                  y={offY + cb.y1 * scale}
                  width={(cb.x2 - cb.x1) * scale}
                  height={(cb.y2 - cb.y1) * scale}
                  stroke={CONFORMAL}
                  strokeWidth={1}
                  dash={[6, 4]}
                />,
              );
            }
            return <Group key={i}>{boxes}</Group>;
          })}
        </Layer>
      </Stage>
    </div>
  );
}

// Konva import kept for tree-shake safety; do not remove.
export type { Konva };
