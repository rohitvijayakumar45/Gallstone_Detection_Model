import { useCallback, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/lib/api";
import type { UploadResponse } from "@/lib/types";

interface Props {
  onUploaded: (u: UploadResponse) => void;
}

export default function UploadDropzone({ onUploaded }: Props) {
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const m = useMutation({
    mutationFn: (file: File) => api.upload(file),
    onSuccess: onUploaded,
  });

  const onFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      m.mutate(files[0]!);
    },
    [m],
  );

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setHover(true);
        }}
        onDragLeave={() => setHover(false)}
        onDrop={(e) => {
          e.preventDefault();
          setHover(false);
          onFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={clsx(
          "cursor-pointer rounded-md border-2 border-dashed px-6 py-8 text-center transition-colors duration-150 ease-clinical",
          hover
            ? "border-accent bg-accent/5 text-text"
            : "border-border bg-surface text-text-2 hover:border-text-3 hover:text-text",
        )}
      >
        <p className="label-caps mb-2">Upload scan</p>
        <p className="text-[13px]">
          Drag a DICOM / JPG / PNG here, or click to browse.
        </p>
        <p className="text-[12px] text-text-3 mt-2 num">
          {m.isPending ? "uploading…" : m.isError ? String(m.error) : "max 20 MB"}
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.bmp,.dcm,.tif,.tiff,.webp"
        className="sr-only"
        onChange={(e) => onFiles(e.target.files)}
      />
    </div>
  );
}
