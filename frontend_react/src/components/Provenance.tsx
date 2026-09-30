import * as Tooltip from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";

interface Props {
  source: string;
  detail?: string | null;
  children: ReactNode;
}

export default function Provenance({ source, detail, children }: Props) {
  return (
    <Tooltip.Provider delayDuration={200}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <span className="underline decoration-dotted decoration-text-3 underline-offset-2 cursor-help">
            {children}
          </span>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            side="top"
            align="center"
            className="z-50 rounded-md border border-border bg-surface px-3 py-2 text-[12px] shadow-2 max-w-xs"
          >
            <div className="label-caps mb-1">Source</div>
            <div className="num text-text">{source}</div>
            {detail ? (
              <div className="text-text-2 mt-1 leading-snug">{detail}</div>
            ) : null}
            <Tooltip.Arrow className="fill-white" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}
