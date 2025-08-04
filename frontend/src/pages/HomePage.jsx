// src/pages/HomePage.jsx - Enhanced with Real Recommendations & Rating
import React, { useEffect, useState } from 'react';
import {
    Box,
    Flex,
    VStack,
    HStack,
    Spinner,
    Text,
    Heading,
    Button,
    useToast,
    Grid,
    GridItem,
    Card,
    CardBody,
    Badge,
    Icon,
    Progress,
    Alert,
    AlertIcon,
    AlertTitle,
    AlertDescription,
} from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { FiMusic, FiRefreshCw, FiSettings, FiTrendingUp, FiStar } from 'react-icons/fi';
import StarRating from '../components/StarRating';

export default function HomePage() {
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [trackDetails, setTrackDetails] = useState({});
    const [userRatings, setUserRatings] = useState({});
    const toast = useToast();
    const navigate = useNavigate();

    // Load recommendations on mount
    useEffect(() => {
        loadRecommendations();
    }, []);

    const loadRecommendations = async () => {
        setLoading(true);
        try {
            // Get recommendations from backend
            const response = await fetch('http://localhost:8000/recommend?alpha=0.7');
            const data = await response.json();
            
            if (data.recommendations && data.recommendations.length > 0) {
                setRecommendations(data.recommendations);
                
                // Fetch track details for each recommendation
                await fetchTrackDetails(data.recommendations);
                
                toast({
                    title: 'Recommendations loaded!',
                    description: `Found ${data.recommendations.length} personalized songs`,
                    status: 'success',
                    duration: 3000,
                    isClosable: true,
                });
            } else {
                setRecommendations([]);
            }
        } catch (error) {
            console.error('Error loading recommendations:', error);
            toast({
                title: 'Failed to load recommendations',
                description: 'Make sure your backend is running on localhost:8000',
                status: 'error',
                duration: 5000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };

    const fetchTrackDetails = async (recs) => {
        const details = {};
        
        // Fetch details for each track
        for (const rec of recs) {
            try {
                const response = await fetch(`http://localhost:8000/track/${rec.track_id}`);
                if (response.ok) {
                    const trackData = await response.json();
                    details[rec.track_id] = trackData;
                }
            } catch (error) {
                console.error(`Failed to fetch details for track ${rec.track_id}:`, error);
                // Fallback data
                details[rec.track_id] = {
                    track_id: rec.track_id,
                    track_name: rec.track_id.substring(0, 10) + '...',
                    artist: 'Unknown Artist',
                    popularity: 0
                };
            }
        }
        
        setTrackDetails(details);
    };

    const refreshRecommendations = async () => {
        setRefreshing(true);
        try {
            await loadRecommendations();
        } finally {
            setRefreshing(false);
        }
    };

    const handleRating = async (trackId, rating) => {
        try {
            // Save rating locally first (optimistic update)
            setUserRatings(prev => ({ ...prev, [trackId]: rating }));
            
            // Submit rating to backend
            const response = await fetch('http://localhost:8000/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    track_id: trackId,
                    rating: rating,
                    user_id: 'default_user'
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            toast({
                title: `Rated ${rating} star${rating !== 1 ? 's' : ''}!`,
                description: 'Your feedback will improve future recommendations',
                status: 'success',
                duration: 3000,
                isClosable: true,
            });
        } catch (error) {
            console.error('Error submitting rating:', error);
            // Revert optimistic update on error
            setUserRatings(prev => {
                const updated = { ...prev };
                delete updated[trackId];
                return updated;
            });
            
            toast({
                title: 'Failed to save rating',
                description: 'Please try again',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        }
    };

    const getScoreColor = (score) => {
        if (score >= 0.8) return 'green';
        if (score >= 0.6) return 'yellow';
        if (score >= 0.4) return 'orange';
        return 'red';
    };

    const getPopularityLevel = (popularity) => {
        if (popularity >= 80) return { text: 'Very Popular', color: 'green' };
        if (popularity >= 60) return { text: 'Popular', color: 'yellow' };
        if (popularity >= 40) return { text: 'Moderate', color: 'orange' };
        if (popularity >= 20) return { text: 'Niche', color: 'blue' };
        return { text: 'Underground', color: 'purple' };
    };

    if (loading) {
        return (
            <Box minH="100vh" bg="black" color="white" p={8}>
                <VStack spacing={6} textAlign="center" mt={20}>
                    <Spinner size="xl" color="purple.500" thickness="4px" />
                    <Text color="gray.400" fontSize="lg">
                        Loading your personalized recommendations...
                    </Text>
                    <Text color="gray.500" fontSize="sm">
                        Using your saved preferences to find the perfect songs
                    </Text>
                </VStack>
            </Box>
        );
    }

    return (
        <Box minH="100vh" bg="black" color="white" p={8}>
            {/* Header */}
            <Flex justify="space-between" align="center" mb={8}>
                <VStack align="start" spacing={2}>
                    <Heading size="xl" color="white">
                        <Icon as={FiMusic} mr={3} />
                        Your Music Recommendations
                    </Heading>
                    <Text color="gray.400" fontSize="md">
                        Powered by Neural Collaborative Filtering & Your Preferences
                    </Text>
                </VStack>
                
                <HStack spacing={3}>
                    <Button
                        variant="outline"
                        colorScheme="purple"
                        onClick={refreshRecommendations}
                        isLoading={refreshing}
                        loadingText="Refreshing..."
                        leftIcon={<FiRefreshCw />}
                    >
                        Refresh
                    </Button>
                    <Button
                        colorScheme="purple"
                        onClick={() => navigate('/profile')}
                        leftIcon={<FiSettings />}
                    >
                        Adjust Preferences
                    </Button>
                </HStack>
            </Flex>

            {/* No Recommendations State */}
            {!loading && recommendations.length === 0 && (
                <Alert status="info" bg="blue.900" borderColor="blue.500" borderWidth={1} borderRadius="md">
                    <AlertIcon />
                    <Box>
                        <AlertTitle>No Recommendations Yet!</AlertTitle>
                        <AlertDescription>
                            Set your music preferences to get personalized song recommendations.
                        </AlertDescription>
                    </Box>
                </Alert>
            )}

            {/* Recommendations Grid */}
            {!loading && recommendations.length > 0 && (
                <VStack spacing={6} align="stretch">
                    {/* Stats Bar */}
                    <Flex justify="space-between" align="center" p={4} bg="gray.900" borderRadius="md">
                        <HStack spacing={6}>
                            <VStack spacing={0}>
                                <Text color="white" fontWeight="bold" fontSize="lg">
                                    {recommendations.length}
                                </Text>
                                <Text color="gray.400" fontSize="sm">
                                    Songs Found
                                </Text>
                            </VStack>
                            <VStack spacing={0}>
                                <Text color="white" fontWeight="bold" fontSize="lg">
                                    {Object.keys(userRatings).length}
                                </Text>
                                <Text color="gray.400" fontSize="sm">
                                    Rated
                                </Text>
                            </VStack>
                            <VStack spacing={0}>
                                <Text color="purple.400" fontWeight="bold" fontSize="lg">
                                    Smart
                                </Text>
                                <Text color="gray.400" fontSize="sm">
                                    AI Mode
                                </Text>
                            </VStack>
                        </HStack>
                        
                        <Badge colorScheme="purple" px={3} py={1} borderRadius="full">
                            <Icon as={FiTrendingUp} mr={1} />
                            Neural CF + Content + Feedback
                        </Badge>
                    </Flex>

                    {/* Recommendations List */}
                    <Grid templateColumns={{ base: '1fr', lg: 'repeat(2, 1fr)' }} gap={4}>
                        {recommendations.map((rec, index) => {
                            const track = trackDetails[rec.track_id] || {};
                            const popularity = getPopularityLevel(track.popularity || 0);
                            const userRating = userRatings[rec.track_id] || 0;
                            
                            return (
                                <GridItem key={rec.track_id}>
                                    <Card bg="gray.900" borderColor="gray.700" borderWidth={1} h="100%">
                                        <CardBody>
                                            <VStack align="stretch" spacing={4}>
                                                {/* Header */}
                                                <Flex justify="space-between" align="start">
                                                    <VStack align="start" spacing={1} flex={1}>
                                                        <Text color="white" fontWeight="bold" fontSize="lg" noOfLines={2}>
                                                            {track.track_name || 'Loading...'}
                                                        </Text>
                                                        <Text color="gray.400" fontSize="md">
                                                            {track.artist || 'Unknown Artist'}
                                                        </Text>
                                                    </VStack>
                                                    <VStack align="end" spacing={1}>
                                                        <Badge colorScheme="purple" variant="subtle">
                                                            #{index + 1}
                                                        </Badge>
                                                    </VStack>
                                                </Flex>

                                                {/* Metrics */}
                                                <HStack justify="space-between">
                                                    <VStack align="start" spacing={1}>
                                                        <Text color="gray.400" fontSize="xs">
                                                            Match Score
                                                        </Text>
                                                        <HStack>
                                                            <Badge colorScheme={getScoreColor(rec.score)} variant="solid">
                                                                {Math.round(rec.score * 100)}%
                                                            </Badge>
                                                            <Progress 
                                                                value={rec.score * 100} 
                                                                colorScheme={getScoreColor(rec.score)}
                                                                size="sm" 
                                                                w="60px"
                                                                bg="gray.700"
                                                                borderRadius="md"
                                                            />
                                                        </HStack>
                                                    </VStack>
                                                    
                                                    <VStack align="end" spacing={1}>
                                                        <Text color="gray.400" fontSize="xs">
                                                            Popularity
                                                        </Text>
                                                        <Badge colorScheme={popularity.color} variant="subtle">
                                                            {popularity.text}
                                                        </Badge>
                                                    </VStack>
                                                </HStack>

                                                {/* Rating Section */}
                                                <VStack spacing={2}>
                                                    <HStack justify="space-between" w="100%">
                                                        <Text color="gray.400" fontSize="sm">
                                                            Rate this song:
                                                        </Text>
                                                        {userRating > 0 && (
                                                            <HStack spacing={1}>
                                                                <Icon as={FiStar} color="yellow.400" />
                                                                <Text color="yellow.400" fontSize="sm" fontWeight="bold">
                                                                    {userRating}/5
                                                                </Text>
                                                            </HStack>
                                                        )}
                                                    </HStack>
                                                    <StarRating
                                                        rating={userRating}
                                                        onChange={(rating) => handleRating(rec.track_id, rating)}
                                                        size="md"
                                                    />
                                                </VStack>
                                            </VStack>
                                        </CardBody>
                                    </Card>
                                </GridItem>
                            );
                        })}
                    </Grid>

                    {/* Footer */}
                    <Flex justify="center" mt={8}>
                        <Text color="gray.500" fontSize="sm" textAlign="center">
                            Recommendations updated based on your preferences • 
                            <Text as="span" color="purple.400" ml={1}>
                                Neural CF Model Active
                            </Text>
                        </Text>
                    </Flex>
                </VStack>
            )}
        </Box>
    );
}