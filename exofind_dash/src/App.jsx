import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ExoFindDashboard from './components/ExoFindDashboard';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ExoFindDashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
