// src/pages/ProfilePage.jsx - Fixed with intuitive sliders
import React, { useState, useEffect } from 'react';
import {
    Box,
    Flex,
    VStack,
    HStack,
    Heading,
    Text,
    Button,
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb,
    Card,
    CardHeader,
    CardBody,
    Badge,
    useToast,
    Spinner,
    Alert,
    AlertIcon,
    AlertTitle,
    AlertDescription,
    Grid,
    GridItem,
    Icon,
    Progress,
    Stat,
    StatLabel,
    StatNumber,
    StatHelpText,
} from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { FiMusic, FiTrendingUp, FiHeart, FiZap, FiSave, FiRefreshCw, FiUser, FiSettings } from 'react-icons/fi';
import { API } from '../api';

// Professional color scheme constants
const COLORS = {
    primary: 'purple',
    secondary: 'blue',
    accent: 'pink',
    success: 'green',
    warning: 'orange',
    error: 'red',
};

// Fixed preference definitions with logical progression
const PREFERENCE_CONFIG = {
    energy: {
        label: 'Energy Level',
        description: 'How energetic you want your music to be',
        icon: FiZap,
        color: COLORS.warning,
        min: 0,
        max: 1,
        step: 0.01,
        labels: {
            left: 'Calm',
            center: 'Moderate',
            right: 'High Energy'
        }
    },
    danceability: {
        label: 'Danceability',
        description: 'How suitable for dancing',
        icon: FiTrendingUp,
        color: COLORS.secondary,
        min: 0,
        max: 1,
        step: 0.01,
        labels: {
            left: 'Not Danceable',
            center: 'Somewhat',
            right: 'Very Danceable'
        }
    },
    valence: {
        label: 'Positivity',
        description: 'How positive/happy the music feels',
        icon: FiHeart,
        color: COLORS.accent,
        min: 0,
        max: 1,
        step: 0.01,
        labels: {
            left: 'Sad/Dark',
            center: 'Neutral',
            right: 'Happy/Bright'
        }
    },
    tempo: {
        label: 'Tempo',
        description: 'Speed and rhythm preference',
        icon: FiMusic,
        color: COLORS.primary,
        min: 0,
        max: 1,
        step: 0.01,
        labels: {
            left: 'Slow',
            center: 'Medium',
            right: 'Fast'
        }
    }
};

export default function ProfilePage() {
    const navigate = useNavigate();
    const toast = useToast();

    // State management
    const [preferences, setPreferences] = useState({
        energy: 0.5,
        danceability: 0.5,
        valence: 0.5,
        tempo: 0.5
    });
    const [pastPreferences, setPastPreferences] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [hasChanges, setHasChanges] = useState(false);

    // Load existing preferences on mount
    useEffect(() => {
        loadPreferences();
    }, []);

    const loadPreferences = async () => {
        setLoading(true);
        try {
            // Use a simple GET request to load preferences
            const response = await fetch('http://localhost:8000/preferences/get');
            const data = await response.json();
            const prefs = data.preferences || data;

            // Extract only the preference values, ignore metadata
            const cleanPrefs = {
                energy: prefs.energy || 0.5,
                danceability: prefs.danceability || 0.5,
                valence: prefs.valence || 0.5,
                tempo: prefs.tempo || 0.5
            };

            setPreferences(cleanPrefs);
            setPastPreferences(cleanPrefs);
            setHasChanges(false);
        } catch (error) {
            console.error('Failed to load preferences:', error);
            toast({
                title: 'Failed to load preferences',
                description: 'Using default values',
                status: 'warning',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };

    const handlePreferenceChange = (key, value) => {
        const newPreferences = { ...preferences, [key]: value };
        setPreferences(newPreferences);

        // Check if there are changes
        const hasChanged = pastPreferences && Object.keys(PREFERENCE_CONFIG).some(
            k => Math.abs(newPreferences[k] - pastPreferences[k]) > 0.01
        );
        setHasChanges(hasChanged);
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            // Validate preferences
            const required = ['energy', 'danceability', 'valence', 'tempo'];
            for (const field of required) {
                if (!(field in preferences)) {
                    throw new Error('Missing required field');
                }
                const value = preferences[field];
                if (typeof value !== 'number' || value < 0 || value > 1) {
                    throw new Error('Invalid preference values');
                }
            }

            // Use simple POST request to save preferences
            const response = await fetch('http://localhost:8000/preferences/set', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(preferences),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            setPastPreferences({ ...preferences });
            setHasChanges(false);

            toast({
                title: 'Preferences saved successfully!',
                description: 'Your music recommendations will be updated',
                status: 'success',
                duration: 4000,
                isClosable: true,
            });
        } catch (error) {
            console.error('Failed to save preferences:', error);
            toast({
                title: 'Failed to save preferences',
                description: error.message || 'Please try again',
                status: 'error',
                duration: 5000,
                isClosable: true,
            });
        } finally {
            setSaving(false);
        }
    };

    const handleReset = () => {
        if (pastPreferences) {
            setPreferences({ ...pastPreferences });
            setHasChanges(false);
        }
    };

    const handleRefresh = () => {
        loadPreferences();
    };

    const getPreferenceScore = (value) => {
        return Math.round(value * 100);
    };

    // Fixed function to get intuitive labels
    const getPreferenceLabel = (key, value) => {
        const config = PREFERENCE_CONFIG[key];
        if (value <= 0.33) return config.labels.left;
        if (value <= 0.66) return config.labels.center;
        return config.labels.right;
    };

    if (loading) {
        return (
            <Box minH="100vh" bg="black" color="white" p={8}>
                <VStack spacing={6} textAlign="center" mt={20}>
                    <Spinner size="xl" color="purple.500" thickness="4px" />
                    <Text color="gray.400" fontSize="lg">
                        Loading your preferences...
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
                        <Icon as={FiUser} mr={3} />
                        Music Preferences
                    </Heading>
                    <Text color="gray.400" fontSize="md">
                        Customize your music recommendations
                    </Text>
                </VStack>

                <HStack spacing={3}>
                    <Button
                        variant="outline"
                        colorScheme="gray"
                        onClick={handleRefresh}
                        leftIcon={<FiRefreshCw />}
                        isLoading={loading}
                    >
                        Refresh
                    </Button>
                    <Button
                        colorScheme="purple"
                        onClick={() => navigate('/')}
                        leftIcon={<FiMusic />}
                    >
                        View Recommendations
                    </Button>
                </HStack>
            </Flex>

            {/* Changes Alert */}
            {hasChanges && (
                <Alert status="info" bg="blue.900" borderColor="blue.500" borderWidth={1} mb={6} borderRadius="md">
                    <AlertIcon />
                    <Box>
                        <AlertTitle>Unsaved Changes</AlertTitle>
                        <AlertDescription>
                            You have unsaved preference changes. Save them to update your recommendations.
                        </AlertDescription>
                    </Box>
                </Alert>
            )}

            <Grid templateColumns={{ base: '1fr', lg: '2fr 1fr' }} gap={8}>
                {/* Main Preferences Panel */}
                <GridItem>
                    <Card bg="gray.900" borderColor="gray.700" borderWidth={1}>
                        <CardHeader>
                            <Flex align="center">
                                <Icon as={FiSettings} color="purple.400" mr={3} />
                                <Heading size="md" color="white">
                                    Preference Settings
                                </Heading>
                            </Flex>
                        </CardHeader>
                        <CardBody>
                            <VStack spacing={8}>
                                {Object.entries(PREFERENCE_CONFIG).map(([key, config]) => (
                                    <Box key={key} w="100%">
                                        <Flex justify="space-between" align="center" mb={4}>
                                            <HStack>
                                                <Icon as={config.icon} color={`${config.color}.400`} />
                                                <VStack align="start" spacing={0}>
                                                    <Text color="white" fontWeight="semibold">
                                                        {config.label}
                                                    </Text>
                                                    <Text color="gray.400" fontSize="sm">
                                                        {config.description}
                                                    </Text>
                                                </VStack>
                                            </HStack>
                                            <VStack align="end" spacing={0}>
                                                <Text color="white" fontWeight="bold" fontSize="lg">
                                                    {getPreferenceScore(preferences[key])}%
                                                </Text>
                                                <Badge colorScheme={config.color} variant="subtle" fontSize="xs">
                                                    {getPreferenceLabel(key, preferences[key])}
                                                </Badge>
                                            </VStack>
                                        </Flex>

                                        <Slider
                                            value={preferences[key]}
                                            min={config.min}
                                            max={config.max}
                                            step={config.step}
                                            onChange={(value) => handlePreferenceChange(key, value)}
                                            colorScheme={config.color}
                                            mb={3}
                                        >
                                            <SliderTrack bg="gray.700" h={2}>
                                                <SliderFilledTrack />
                                            </SliderTrack>
                                            <SliderThumb boxSize={6} boxShadow="lg">
                                                <Box color={`${config.color}.500`} as={config.icon} />
                                            </SliderThumb>
                                        </Slider>

                                        {/* Fixed slider labels - properly positioned */}
                                        <Flex justify="space-between" align="center" mt={2}>
                                            <Text color="gray.500" fontSize="xs" fontWeight="medium">
                                                {config.labels.left}
                                            </Text>
                                            <Text
                                                color="gray.400"
                                                fontSize="xs"
                                                fontWeight="medium"
                                                bg="gray.800"
                                                px={2}
                                                py={1}
                                                borderRadius="md"
                                            >
                                                {config.labels.center}
                                            </Text>
                                            <Text color="gray.500" fontSize="xs" fontWeight="medium">
                                                {config.labels.right}
                                            </Text>
                                        </Flex>
                                    </Box>
                                ))}
                            </VStack>
                        </CardBody>
                    </Card>

                    {/* Action Buttons */}
                    <HStack spacing={4} mt={6} justify="center">
                        <Button
                            variant="outline"
                            colorScheme="gray"
                            onClick={handleReset}
                            isDisabled={!hasChanges}
                            size="lg"
                        >
                            Reset Changes
                        </Button>
                        <Button
                            colorScheme="purple"
                            onClick={handleSave}
                            isLoading={saving}
                            loadingText="Saving..."
                            leftIcon={<FiSave />}
                            size="lg"
                            isDisabled={!hasChanges}
                        >
                            Save Preferences
                        </Button>
                    </HStack>
                </GridItem>

                {/* Sidebar - Current Profile Stats */}
                <GridItem>
                    <VStack spacing={6}>
                        {/* Current Profile Summary */}
                        <Card bg="gray.900" borderColor="gray.700" borderWidth={1} w="100%">
                            <CardHeader>
                                <Heading size="md" color="white">
                                    Your Music Profile
                                </Heading>
                            </CardHeader>
                            <CardBody>
                                <VStack spacing={4}>
                                    {Object.entries(PREFERENCE_CONFIG).map(([key, config]) => (
                                        <Stat key={key}>
                                            <StatLabel color="gray.400" fontSize="sm">
                                                <Icon as={config.icon} mr={2} />
                                                {config.label}
                                            </StatLabel>
                                            <StatNumber color="white" fontSize="lg">
                                                {getPreferenceScore(preferences[key])}%
                                            </StatNumber>
                                            <StatHelpText color="gray.500" fontSize="xs">
                                                {getPreferenceLabel(key, preferences[key])}
                                            </StatHelpText>
                                            <Progress
                                                value={preferences[key] * 100}
                                                colorScheme={config.color}
                                                size="sm"
                                                bg="gray.700"
                                                borderRadius="md"
                                            />
                                        </Stat>
                                    ))}
                                </VStack>
                            </CardBody>
                        </Card>

                        {/* Past Preferences Comparison */}
                        {pastPreferences && hasChanges && (
                            <Card bg="gray.900" borderColor="blue.500" borderWidth={1} w="100%">
                                <CardHeader>
                                    <Heading size="sm" color="blue.400">
                                        Changes Preview
                                    </Heading>
                                </CardHeader>
                                <CardBody>
                                    <VStack spacing={3}>
                                        {Object.entries(PREFERENCE_CONFIG).map(([key, config]) => {
                                            const change = preferences[key] - pastPreferences[key];
                                            const hasChange = Math.abs(change) > 0.01;

                                            if (!hasChange) return null;

                                            return (
                                                <Flex key={key} justify="space-between" align="center" w="100%">
                                                    <Text color="gray.400" fontSize="sm">
                                                        {config.label}
                                                    </Text>
                                                    <Badge
                                                        colorScheme={change > 0 ? 'green' : 'red'}
                                                        variant="subtle"
                                                    >
                                                        {change > 0 ? '+' : ''}{Math.round(change * 100)}%
                                                    </Badge>
                                                </Flex>
                                            );
                                        })}
                                    </VStack>
                                </CardBody>
                            </Card>
                        )}

                        {/* Tips Card */}
                        <Card bg="purple.900" borderColor="purple.500" borderWidth={1} w="100%">
                            <CardHeader>
                                <Heading size="sm" color="purple.300">
                                    💡 Pro Tips
                                </Heading>
                            </CardHeader>
                            <CardBody>
                                <VStack spacing={2} align="start">
                                    <Text color="purple.200" fontSize="sm">
                                        • Slide right for more intense preferences
                                    </Text>
                                    <Text color="purple.200" fontSize="sm">
                                        • Mix different settings for unique recommendations
                                    </Text>
                                    <Text color="purple.200" fontSize="sm">
                                        • Try extreme settings to discover new genres
                                    </Text>
                                    <Text color="purple.200" fontSize="sm">
                                        • Save changes to update your recommendations
                                    </Text>
                                </VStack>
                            </CardBody>
                        </Card>
                    </VStack>
                </GridItem>
            </Grid>
        </Box>
    );
}