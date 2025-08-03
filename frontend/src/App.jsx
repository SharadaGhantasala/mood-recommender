// src/App.js - FIXED VERSION
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Flex, Box } from '@chakra-ui/react';
import NavBar from './components/NavBar'; // Remove the curly braces
import HomePage from './pages/HomePage';
import ProfilePage from './pages/ProfilePage';
import StatsPage from './pages/StatsPage';

function App() {
  return (
    <BrowserRouter>
      <Flex minH="100vh" bg="black">
        <NavBar />
        <Box
          flex="1"
          ml="72px"
          bg="black"
          minH="100vh"
        >
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/stats" element={<StatsPage />} />
          </Routes>
        </Box>
      </Flex>
    </BrowserRouter>
  );
}

export default App;