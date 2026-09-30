import { useParams } from "react-router-dom";

export default function ComparePage() {
  const { idA, idB } = useParams();
  return (
    <div className="grid grid-cols-2 h-[calc(100vh-56px-40px)]">
      <div className="border-r border-border p-6">
        <p className="label-caps mb-2">Scan A</p>
        <p className="num text-text-2 text-[13px]">{idA ?? "—"}</p>
      </div>
      <div className="p-6">
        <p className="label-caps mb-2">Scan B</p>
        <p className="num text-text-2 text-[13px]">{idB ?? "—"}</p>
      </div>
    </div>
  );
}
