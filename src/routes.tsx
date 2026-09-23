import { Navigate, Route, Routes } from "react-router-dom";
import { AboutPage } from "@/pages/AboutPage";
import { ChatModelsPage } from "@/pages/ChatModelsPage";
import { DashboardPage } from "@/pages/DashboardPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/about" element={<AboutPage />} />
      <Route path="/chat" element={<ChatModelsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
