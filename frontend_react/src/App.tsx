import { Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import LandingPage from "./pages/LandingPage";
import WorkstationPage from "./pages/WorkstationPage";
import BenchmarksPage from "./pages/BenchmarksPage";
import ComparePage from "./pages/ComparePage";
import AuditPage from "./pages/AuditPage";
import NotFoundPage from "./pages/NotFoundPage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/study/:scanId" element={<WorkstationPage />} />
        <Route path="/study" element={<WorkstationPage />} />
        <Route path="/benchmarks" element={<BenchmarksPage />} />
        <Route path="/compare/:idA/:idB" element={<ComparePage />} />
        <Route path="/audit" element={<AuditPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </AppShell>
  );
}
