// src/pages/StatsPage.jsx - Comprehensive Analytics Dashboard
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
    Card,
    CardBody,
    CardHeader,
    Badge,
    Progress,
    Stat,
    StatLabel,
    StatNumber,
    StatHelpText,
    StatArrow,
    Table,
    Thead,
    Tbody,
    Tr,
    Th,
    Td,
    TableContainer,
    Icon,
    Tabs,
    TabList,
    TabPanels,
    Tab,
    TabPanel
} from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { 
    FiTrendingUp, 
    FiTrendingDown, 
    FiActivity, 
    FiTarget, 
    FiUsers, 
    FiStar,
    FiRefreshCw,
    FiBarChart,
    FiPieChart,
    FiSettings,
    FiCheck,
    FiX,
    FiClock
} from 'react-icons/fi';

export default function StatsPage() {
    const [overview, setOverview] = useState(null);
    const [contextualBandits, setContextualBandits] = useState(null);
    const [abTesting, setAbTesting] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const toast = useToast();
    const navigate = useNavigate();

    useEffect(() => {
        loadAllAnalytics();
        // Auto-refresh every 30 seconds for real-time feel
        const interval = setInterval(loadAllAnalytics, 30000);
        return () => clearInterval(interval);
    }, []);

    const loadAllAnalytics = async () => {
        setLoading(true);
        try {
            const [overviewRes, banditRes, abRes] = await Promise.all([
                fetch('http://localhost:8000/analytics/overview'),
                fetch('http://localhost:8000/analytics/contextual-bandits'),
                fetch('http://localhost:8000/analytics/ab-testing')
            ]);

            if (overviewRes.ok && banditRes.ok && abRes.ok) {
                const [overviewData, banditData, abData] = await Promise.all([
                    overviewRes.json(),
                    banditRes.json(),
                    abRes.json()
                ]);

                setOverview(overviewData);
                setContextualBandits(banditData);
                setAbTesting(abData);
            } else {
                throw new Error('Failed to fetch analytics data');
            }
        } catch (error) {
            console.error('Error loading analytics:', error);
            toast({
                title: 'Failed to load analytics',
                description: 'Make sure your backend is running on localhost:8000',
                status: 'error',
                duration: 5000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };

    const refreshAnalytics = async () => {
        setRefreshing(true);
        try {
            await loadAllAnalytics();
            toast({
                title: 'Analytics refreshed!',
                description: 'Latest data has been loaded',
                status: 'success',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setRefreshing(false);
        }
    };

    if (loading) {
        return (
            <Box minH="100vh" bg="black" color="white" p={8}>
                <VStack spacing={6} textAlign="center" mt={20}>
                    <Spinner size="xl" color="purple.500" thickness="4px" />
                    <Text color="gray.400" fontSize="lg">
                        Loading analytics dashboard...
                    </Text>
                    <Text color="gray.500" fontSize="sm">
                        Fetching real-time contextual bandits and A/B testing data
                    </Text>
                </VStack>
            </Box>
        );
    }

    const getStrategyColor = (strategy) => {
        const colors = {
            hybrid: 'purple',
            collaborative_filtering: 'blue',
            content_based: 'green',
            popularity: 'orange'
        };
        return colors[strategy] || 'gray';
    };

    const formatStrategy = (strategy) => {
        const names = {
            hybrid: 'Hybrid (CF + Content)',
            collaborative_filtering: 'Collaborative Filtering',
            content_based: 'Content-Based',
            popularity: 'Popularity-Based'
        };
        return names[strategy] || strategy;
    };

    return (
        <Box minH="100vh" bg="black" color="white" p={8}>
            {/* Header */}
            <Flex justify="space-between" align="center" mb={8}>
                <VStack align="start" spacing={2}>
                    <Heading size="xl" color="white">
                        <Icon as={FiBarChart} mr={3} />
                        Analytics Dashboard
                    </Heading>
                    <Text color="gray.400" fontSize="md">
                        Real-time contextual bandits and A/B testing insights for Neural CF
                    </Text>
                </VStack>
                
                <HStack spacing={3}>
                    <Button
                        variant="outline"
                        colorScheme="purple"
                        onClick={refreshAnalytics}
                        isLoading={refreshing}
                        loadingText="Refreshing..."
                        leftIcon={<FiRefreshCw />}
                    >
                        Refresh Data
                    </Button>
                    <Button
                        colorScheme="purple"
                        onClick={() => navigate('/profile')}
                        leftIcon={<FiSettings />}
                    >
                        Settings
                    </Button>
                </HStack>
            </Flex>

            <Tabs variant="enclosed" colorScheme="purple">
                <TabList mb={6}>
                    <Tab _selected={{ color: 'white', bg: 'purple.600' }}>Overview</Tab>
                    <Tab _selected={{ color: 'white', bg: 'purple.600' }}>Contextual Bandits</Tab>
                    <Tab _selected={{ color: 'white', bg: 'purple.600' }}>A/B Testing</Tab>
                </TabList>

                <TabPanels>
                    {/* Overview Tab */}
                    <TabPanel p={0}>
                        {overview && (
                            <VStack spacing={6} align="stretch">
                                {/* System Overview Stats */}
                                <Card bg="gray.900" borderColor="gray.700" borderWidth={1}>
                                    <CardHeader>
                                        <Heading size="md" color="white">
                                            <Icon as={FiActivity} mr={2} />
                                            System Overview
                                        </Heading>
                                    </CardHeader>
                                    <CardBody>
                                        <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }} gap={6}>
                                            <Stat>
                                                <StatLabel color="gray.400">Total Users</StatLabel>
                                                <StatNumber color="white">{overview.overview.total_users}</StatNumber>
                                                <StatHelpText color="gray.500">
                                                    <Icon as={FiUsers} mr={1} />
                                                    Active profiles
                                                </StatHelpText>
                                            </Stat>
                                            <Stat>
                                                <StatLabel color="gray.400">Total Ratings</StatLabel>
                                                <StatNumber color="white">{overview.overview.total_ratings}</StatNumber>
                                                <StatHelpText color="gray.500">
                                                    <Icon as={FiStar} mr={1} />
                                                    User feedback
                                                </StatHelpText>
                                            </Stat>
                                            <Stat>
                                                <StatLabel color="gray.400">Avg Rating</StatLabel>
                                                <StatNumber color="white">{overview.overview.avg_rating}/5</StatNumber>
                                                <StatHelpText color="green.400">
                                                    <StatArrow type="increase" />
                                                    User satisfaction
                                                </StatHelpText>
                                            </Stat>
                                            <Stat>
                                                <StatLabel color="gray.400">Active Strategies</StatLabel>
                                                <StatNumber color="white">{overview.overview.active_strategies}</StatNumber>
                                                <StatHelpText color="gray.500">
                                                    <Icon as={FiTarget} mr={1} />
                                                    Recommendation algorithms
                                                </StatHelpText>
                                            </Stat>
                                        </Grid>
                                    </CardBody>
                                </Card>

                                {/* Rating Distribution */}
                                <Card bg="gray.900" borderColor="gray.700" borderWidth={1}>
                                    <CardHeader>
                                        <Heading size="md" color="white">
                                            <Icon as={FiPieChart} mr={2} />
                                            Rating Distribution
                                        </Heading>
                                    </CardHeader>
                                    <CardBody>
                                        <Grid templateColumns="repeat(5, 1fr)" gap={4}>
                                            {Object.entries(overview.overview.rating_distribution).map(([rating, count]) => {
                                                const total = overview.overview.total_ratings;
                                                const percentage = total > 0 ? (count / total * 100) : 0;
                                                return (
                                                    <VStack key={rating} spacing={2}>
                                                        <Text color="gray.400" fontSize="sm">{rating} ⭐</Text>
                                                        <Progress 
                                                            value={percentage} 
                                                            colorScheme={rating >= '4' ? 'green' : rating >= '3' ? 'yellow' : 'red'}
                                                            size="lg" 
                                                            height="60px"
                                                            orientation="vertical"
                                                            bg="gray.700"
                                                            borderRadius="md"
                                                        />
                                                        <Text color="white" fontSize="sm" fontWeight="bold">{count}</Text>
                                                        <Text color="gray.500" fontSize="xs">{percentage.toFixed(1)}%</Text>
                                                    </VStack>
                                                );
                                            })}
                                        </Grid>
                                    </CardBody>
                                </Card>

                                {/* Strategy Performance Overview */}
                                <Card bg="gray.900" borderColor="gray.700" borderWidth={1}>
                                    <CardHeader>
                                        <Heading size="md" color="white">
                                            <Icon as={FiTarget} mr={2} />
                                            Strategy Performance
                                        </Heading>
                                        <Badge colorScheme="purple" ml={3}>
                                            Best: {formatStrategy(overview.contextual_bandits.best_performing_strategy)}
                                        </Badge>
                                    </CardHeader>
                                    <CardBody>
                                        <Grid templateColumns={{ base: '1fr', lg: 'repeat(2, 1fr)' }} gap={4}>
                                            {Object.entries(overview.contextual_bandits.strategy_performance).map(([strategy, performance]) => (
                                                <Card key={strategy} bg="gray.800" borderColor="gray.600" borderWidth={1}>
                                                    <CardBody>
                                                        <VStack align="stretch" spacing={3}>
                                                            <HStack justify="space-between">
                                                                <Text color="white" fontWeight="bold">
                                                                    {formatStrategy(strategy)}
                                                                </Text>
                                                                <Badge colorScheme={getStrategyColor(strategy)}>
                                                                    CTR: {(performance.click_through_rate * 100).toFixed(1)}%
                                                                </Badge>
                                                            </HStack>
                                                            <HStack justify="space-between">
                                                                <VStack align="start" spacing={1}>
                                                                    <Text color="gray.400" fontSize="sm">Impressions</Text>
                                                                    <Text color="white" fontWeight="bold">{performance.impressions.toLocaleString()}</Text>
                                                                </VStack>
                                                                <VStack align="center" spacing={1}>
                                                                    <Text color="gray.400" fontSize="sm">Clicks</Text>
                                                                    <Text color="white" fontWeight="bold">{performance.clicks}</Text>
                                                                </VStack>
                                                                <VStack align="end" spacing={1}>
                                                                    <Text color="gray.400" fontSize="sm">Avg Rating</Text>
                                                                    <Text color="white" fontWeight="bold">{performance.avg_rating}/5</Text>
                                                                </VStack>
                                                            </HStack>
                                                        </VStack>
                                                    </CardBody>
                                                </Card>
                                            ))}
                                        </Grid>
                                    </CardBody>
                                </Card>
                            </VStack>
                        )}
                    </TabPanel>

                    {/* Contextual Bandits Tab */}
                    <TabPanel p={0}>
                        {contextualBandits && (
                            <VStack spacing={6} align="stretch">
                                {/* Current Performance */}
                                <Card bg="gray.900" borderColor="gray.700" borderWidth={1}>
                                    <CardHeader>
                                        <Heading size="md" color="white">
                                            <Icon as={FiTrendingUp} mr={2} />
                                            Current Strategy Performance
                                        </Heading>
                                        <Badge colorScheme="green" ml={3}>
                                            Best: {formatStrategy(contextualBandits.best_strategy)}
                                        </Badge>
                                    </CardHeader>
                                    <CardBody>
                                        <TableContainer>
                                            <Table variant="simple" size="sm">
                                                <Thead>
                                                    <Tr>
                                                        <Th color="gray.400">Strategy</Th>
                                                        <Th color="gray.400">CTR (3-day avg)</Th>
                                                        <Th color="gray.400">Satisfaction</Th>
                                                        <Th color="gray.400">Trend</Th>
                                                        <Th color="gray.400">Confidence Interval</Th>
                                                    </Tr>
                                                </Thead>
                                                <Tbody>
                                                    {Object.entries(contextualBandits.current_performance).map(([strategy, perf]) => (
                                                        <Tr key={strategy}>
                                                            <Td>
                                                                <Badge colorScheme={getStrategyColor(strategy)}>
                                                                    {formatStrategy(strategy)}
                                                                </Badge>
                                                            </Td>
                                                            <Td color="white" fontWeight="bold">
                                                                {(perf.avg_ctr * 100).toFixed(1)}%
                                                            </Td>
                                                            <Td color="white">{perf.avg_satisfaction}/5</Td>
                                                            <Td>
                                                                <HStack>
                                                                    <Icon 
                                                                        as={perf.trend === 'increasing' ? FiTrendingUp : FiTrendingDown} 
                                                                        color={perf.trend === 'increasing' ? 'green.400' : 'red.400'}
                                                                    />
                                                                    <Text 
                                                                        color={perf.trend === 'increasing' ? 'green.400' : 'red.400'}
                                                                        fontSize="sm"
                                                                    >
                                                                        {perf.trend}
                                                                    </Text>
                                                                </HStack>
                                                            </Td>
                                                            <Td color="gray.400" fontSize="sm">
                                                                [{perf.confidence_interval[0].toFixed(3)}, {perf.confidence_interval[1].toFixed(3)}]
                                                            </Td>
                                                        </Tr>
                                                    ))}
                                                </Tbody>
                                            </Table>
                                        </TableContainer>
                                    </CardBody>
                                </Card>

                                {/* Statistical Significance */}
                                <Card bg="gray.900" borderColor="gray.700" borderWidth={1}>
                                    <CardHeader>
                                        <Heading size="md" color="white">
                                            <Icon as={FiCheck} mr={2} />
                                            Statistical Significance Tests
                                        </Heading>
                                    </CardHeader>
                                    <CardBody>
                                        <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap={4}>
                                            {Object.entries(contextualBandits.statistical_significance).map(([comparison, result]) => (
                                                <Card key={comparison} bg={result.significant ? 'green.900' : 'gray.800'} borderColor={result.significant ? 'green.500' : 'gray.600'} borderWidth={1}>
                                                    <CardBody>
                                                        <VStack spacing={2}>
                                                            <HStack>
                                                                <Icon as={result.significant ? FiCheck : FiX} color={result.significant ? 'green.400' : 'red.400'} />
                                                                <Text color="white" fontSize="sm" fontWeight="bold">
                                                                    {comparison.replace('_', ' vs ').toUpperCase()}
                                                                </Text>
                                                            </HStack>
                                                            <Text color="gray.400" fontSize="xs">p-value: {result.p_value}</Text>
                                                            <Badge colorScheme={result.significant ? 'green' : 'red'} size="sm">
                                                                {result.significant ? 'Significant' : 'Not Significant'}
                                                            </Badge>
                                                        </VStack>
                                                    </CardBody>
                                                </Card>
                                            ))}
                                        </Grid>
                                    </CardBody>
                                </Card>
                            </VStack>
                        )}
                    </TabPanel>

                    {/* A/B Testing Tab */}
                    <TabPanel p={0}>
                        {abTesting && (
                            <VStack spacing={6} align="stretch">
                                {/* Experiment Summary */}
                                <Card bg="gray.900" borderColor="gray.700" borderWidth={1}>
                                    <CardHeader>
                                        <Heading size="md" color="white">
                                            <Icon as={FiActivity} mr={2} />
                                            A/B Testing Summary
                                        </Heading>
                                    </CardHeader>
                                    <CardBody>
                                        <Grid templateColumns={{ base: '1fr', md: 'repeat(4, 1fr)' }} gap={6}>
                                            <Stat>
                                                <StatLabel color="gray.400">Total Experiments</StatLabel>
                                                <StatNumber color="white">{abTesting.summary.total_experiments}</StatNumber>
                                            </Stat>
                                            <Stat>
                                                <StatLabel color="gray.400">Active Tests</StatLabel>
                                                <StatNumber color="white">{abTesting.summary.active_experiments}</StatNumber>
                                                <StatHelpText color="green.400">
                                                    <Icon as={FiClock} mr={1} />
                                                    Running now
                                                </StatHelpText>
                                            </Stat>
                                            <Stat>
                                                <StatLabel color="gray.400">Significant Results</StatLabel>
                                                <StatNumber color="white">{abTesting.summary.significant_results}</StatNumber>
                                                <StatHelpText color="blue.400">
                                                    <Icon as={FiCheck} mr={1} />
                                                    Statistically valid
                                                </StatHelpText>
                                            </Stat>
                                            <Stat>
                                                <StatLabel color="gray.400">Avg Effect Size</StatLabel>
                                                <StatNumber color="white">{(abTesting.summary.avg_effect_size * 100).toFixed(1)}%</StatNumber>
                                                <StatHelpText color="purple.400">
                                                    <StatArrow type="increase" />
                                                    Impact magnitude
                                                </StatHelpText>
                                            </Stat>
                                        </Grid>
                                    </CardBody>
                                </Card>

                                {/* Individual Experiments */}
                                {Object.entries(abTesting.experiments).map(([experimentId, experiment]) => (
                                    <Card key={experimentId} bg="gray.900" borderColor="gray.700" borderWidth={1}>
                                        <CardHeader>
                                            <HStack justify="space-between">
                                                <VStack align="start" spacing={1}>
                                                    <Heading size="md" color="white">{experiment.name}</Heading>
                                                    <Text color="gray.400" fontSize="sm">
                                                        {experiment.start_date} {experiment.end_date ? `- ${experiment.end_date}` : '(ongoing)'}
                                                    </Text>
                                                </VStack>
                                                <VStack align="end" spacing={1}>
                                                    <Badge 
                                                        colorScheme={experiment.status === 'completed' ? 'green' : 'blue'}
                                                        size="lg"
                                                    >
                                                        {experiment.status}
                                                    </Badge>
                                                    {experiment.winner && (
                                                        <Badge colorScheme="purple" size="sm">
                                                            Winner: {experiment.variants[experiment.winner].name}
                                                        </Badge>
                                                    )}
                                                </VStack>
                                            </HStack>
                                        </CardHeader>
                                        <CardBody>
                                            <VStack spacing={4}>
                                                {/* Variants Comparison */}
                                                <Grid templateColumns={{ base: '1fr', lg: 'repeat(2, 1fr)' }} gap={4} w="100%">
                                                    {Object.entries(experiment.variants).map(([variantId, variant]) => (
                                                        <Card 
                                                            key={variantId} 
                                                            bg={experiment.winner === variantId ? 'green.900' : 'gray.800'} 
                                                            borderColor={experiment.winner === variantId ? 'green.500' : 'gray.600'} 
                                                            borderWidth={1}
                                                        >
                                                            <CardBody>
                                                                <VStack align="stretch" spacing={3}>
                                                                    <HStack justify="space-between">
                                                                        <Text color="white" fontWeight="bold" fontSize="sm">
                                                                            {variant.name}
                                                                        </Text>
                                                                        {experiment.winner === variantId && (
                                                                            <Badge colorScheme="green" size="sm">Winner</Badge>
                                                                        )}
                                                                    </HStack>
                                                                    <Grid templateColumns="repeat(2, 1fr)" gap={3}>
                                                                        <VStack spacing={1}>
                                                                            <Text color="gray.400" fontSize="xs">Users</Text>
                                                                            <Text color="white" fontWeight="bold">{variant.users}</Text>
                                                                        </VStack>
                                                                        <VStack spacing={1}>
                                                                            <Text color="gray.400" fontSize="xs">Conversions</Text>
                                                                            <Text color="white" fontWeight="bold">{variant.conversions}</Text>
                                                                        </VStack>
                                                                        <VStack spacing={1}>
                                                                            <Text color="gray.400" fontSize="xs">CVR</Text>
                                                                            <Text color="white" fontWeight="bold">
                                                                                {(variant.conversion_rate * 100).toFixed(1)}%
                                                                            </Text>
                                                                        </VStack>
                                                                        <VStack spacing={1}>
                                                                            <Text color="gray.400" fontSize="xs">Rating</Text>
                                                                            <Text color="white" fontWeight="bold">{variant.avg_rating}/5</Text>
                                                                        </VStack>
                                                                    </Grid>
                                                                </VStack>
                                                            </CardBody>
                                                        </Card>
                                                    ))}
                                                </Grid>

                                                {/* Statistical Results */}
                                                <Card bg="gray.800" borderColor="gray.600" borderWidth={1} w="100%">
                                                    <CardBody>
                                                        <HStack justify="space-between" wrap="wrap" spacing={4}>
                                                            <VStack spacing={1}>
                                                                <Text color="gray.400" fontSize="xs">P-Value</Text>
                                                                <Text color="white" fontWeight="bold">{experiment.statistical_results.p_value}</Text>
                                                            </VStack>
                                                            <VStack spacing={1}>
                                                                <Text color="gray.400" fontSize="xs">Confidence</Text>
                                                                <Text color="white" fontWeight="bold">
                                                                    {(experiment.statistical_results.confidence_level * 100)}%
                                                                </Text>
                                                            </VStack>
                                                            <VStack spacing={1}>
                                                                <Text color="gray.400" fontSize="xs">Effect Size</Text>
                                                                <Text color="white" fontWeight="bold">
                                                                    {(experiment.statistical_results.effect_size * 100).toFixed(1)}%
                                                                </Text>
                                                            </VStack>
                                                            <VStack spacing={1}>
                                                                <Text color="gray.400" fontSize="xs">Statistical Power</Text>
                                                                <Text color="white" fontWeight="bold">
                                                                    {(experiment.statistical_results.power * 100).toFixed(0)}%
                                                                </Text>
                                                            </VStack>
                                                            <VStack spacing={1}>
                                                                <Text color="gray.400" fontSize="xs">Significant</Text>
                                                                <Badge 
                                                                    colorScheme={experiment.statistical_results.significant ? 'green' : 'red'}
                                                                    size="sm"
                                                                >
                                                                    {experiment.statistical_results.significant ? 'Yes' : 'No'}
                                                                </Badge>
                                                            </VStack>
                                                        </HStack>
                                                    </CardBody>
                                                </Card>
                                            </VStack>
                                        </CardBody>
                                    </Card>
                                ))}

                                {/* Recommendations */}
                                <Card bg="blue.900" borderColor="blue.500" borderWidth={1}>
                                    <CardHeader>
                                        <Heading size="md" color="white">
                                            <Icon as={FiTarget} mr={2} />
                                            Recommendations
                                        </Heading>
                                    </CardHeader>
                                    <CardBody>
                                        <VStack align="stretch" spacing={3}>
                                            {abTesting.recommendations.map((rec, index) => (
                                                <HStack key={index} spacing={3}>
                                                    <Badge colorScheme="blue" minW="20px">{index + 1}</Badge>
                                                    <Text color="white" fontSize="sm">{rec}</Text>
                                                </HStack>
                                            ))}
                                        </VStack>
                                    </CardBody>
                                </Card>
                            </VStack>
                        )}
                    </TabPanel>
                </TabPanels>
            </Tabs>

            {/* Footer */}
            <Flex justify="center" mt={8}>
                <Text color="gray.500" fontSize="sm" textAlign="center">
                    Analytics updated in real-time • 
                    <Text as="span" color="purple.400" ml={1}>
                        Neural CF Analytics Engine Active
                    </Text>
                </Text>
            </Flex>
        </Box>
    );
}