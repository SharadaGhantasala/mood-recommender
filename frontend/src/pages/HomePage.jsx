// src/pages/HomePage.jsx - FIXED VERSION
import React, { useEffect, useState } from 'react';
import {
    Box,
    Flex,
    VStack,
    Spinner,
    Text,
    Heading,
    Button,
    useToast
} from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import RecommendationCard from '../components/RecommendationCard';
import { API } from '../api';

export default function HomePage() {
    const [recs, setRecs] = useState([]);
    const [loading, setLoading] = useState(true);
    const toast = useToast();
    const navigate = useNavigate();
    const userId = 'demo_user_1';

    // Move loadRecommendations inside useEffect to fix the warning
    useEffect(() => {
        const loadRecommendations = async () => {
            setLoading(true);
            try {
                console.log('Fetching recommendations for:', userId);
                const data = await API.recommend({
                    user_id: userId,
                    n_recommendations: 8,
                    mood: 'happy',
                    device_type: 'desktop'
                });
                console.log('Received recommendations:', data);
                setRecs(data);
            } catch (err) {
                console.error('Error loading recommendations:', err);
                toast({
                    title: 'Could not load recommendations',
                    description: 'Check if backend is running on localhost:8000',
                    status: 'error',
                    duration: 5000,
                    isClosable: true,
                });
            } finally {
                setLoading(false);
            }
        };

        loadRecommendations();
    }, [userId, toast]); // Add dependencies

    // Create a separate refresh function for the button
    const refreshRecommendations = async () => {
        setLoading(true);
        try {
            console.log('Refreshing recommendations for:', userId);
            const data = await API.recommend({
                user_id: userId,
                n_recommendations: 8,
                mood: 'happy',
                device_type: 'desktop'
            });
            console.log('Received recommendations:', data);
            setRecs(data);
        } catch (err) {
            console.error('Error loading recommendations:', err);
            toast({
                title: 'Could not load recommendations',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };

    // Rating callback
    const handleFeedback = async (trackId, rating) => {
        try {
            console.log('Submitting feedback:', { user_id: userId, track_id: trackId, rating });
            await API.feedback({
                user_id: userId,
                track_id: trackId,
                rating: rating,
                context: {
                    mood: 'happy',
                    device_type: 'desktop'
                }
            });
            toast({
                title: `Rated ${rating} star${rating > 1 ? 's' : ''}!`,
                status: 'success',
                duration: 3000,
                isClosable: true,
            });
        } catch (err) {
            console.error('Error submitting feedback:', err);
            toast({
                title: 'Could not submit rating',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        }
    };

    return (
        <Box
            minH="100vh"
            bg="black"
            color="white"
            p={8}
        >
            {/* Header */}
            <Flex justify="space-between" align="center" mb={8}>
                <Heading size="xl" color="white">
                    Home
                </Heading>
                {recs.length > 0 && (
                    <Flex gap={3}>
                        <Button
                            variant="outline"
                            colorScheme="purple"
                            onClick={refreshRecommendations}
                            isLoading={loading}
                        >
                            Refresh
                        </Button>
                        <Button
                            colorScheme="purple"
                            onClick={() => navigate('/profile')}
                        >
                            Adjust Preferences
                        </Button>
                    </Flex>
                )}
            </Flex>

            {/* Loading State */}
            {loading && (
                <VStack spacing={6} textAlign="center" mt={20}>
                    <Spinner size="xl" color="purple.500" thickness="4px" />
                    <Text color="gray.400" fontSize="lg">
                        Loading your personalized recommendations...
                    </Text>
                </VStack>
            )}

            {/* No Recommendations State */}
            {!loading && !recs.length && (
                <VStack spacing={6} textAlign="center" mt={20}>
                    <Text color="gray.400" fontSize="lg">
                        No recommendations yet—set your preferences first!
                    </Text>
                    <VStack spacing={3}>
                        <Button
                            colorScheme="purple"
                            size="lg"
                            onClick={() => navigate('/profile')}
                        >
                            Set Your Preferences
                        </Button>
                        <Button
                            variant="outline"
                            colorScheme="purple"
                            onClick={refreshRecommendations}
                        >
                            Try Loading Again
                        </Button>
                    </VStack>
                </VStack>
            )}

            {/* Recommendations List */}
            {!loading && recs.length > 0 && (
                <VStack spacing={4} align="stretch">
                    <Text color="gray.400" fontSize="md" mb={2}>
                        Powered by Enhanced Neural CF • {recs.length} personalized songs
                    </Text>
                    {recs.map((rec) => (
                        <RecommendationCard
                            key={rec.track_id}
                            rec={rec}
                            onFeedback={handleFeedback}
                        />
                    ))}
                </VStack>
            )}
        </Box>
    );
}