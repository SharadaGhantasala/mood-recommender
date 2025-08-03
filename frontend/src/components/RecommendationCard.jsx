import React, { useState } from 'react';
import { Box, Heading, Text, Badge, HStack } from '@chakra-ui/react';
import StarRating from './StarRating';

/**
 * Props:
 * - rec: { track_id, track_name, artist, popularity, prediction_score, confidence, strategy }
 * - onFeedback: (track_id: string, rating: number) => void
 */
export default function RecommendationCard({ rec, onFeedback }) {
    const { track_id, track_name, artist, popularity, prediction_score, confidence, strategy } = rec;
    const [userRating, setUserRating] = useState(0);

    const handleRating = (stars) => {
        setUserRating(stars);
        onFeedback(track_id, stars);
    };

    return (
        <Box
            p={4}
            bg="purple.850"
            border="1px solid"
            borderColor="purple.800"
            rounded="md"
            position="relative"
        >
            <HStack justify="space-between">
                <Box>
                    <Heading size="md" color="white">
                        {track_name}
                    </Heading>
                    <Text fontSize="sm" color="gray.400">
                        {artist}
                    </Text>
                </Box>
                <Badge colorScheme="yellow">Popularity {popularity}</Badge>
            </HStack>

            <HStack mt={2} spacing={2}>
                <Badge colorScheme="purple">Score: {prediction_score.toFixed(2)}</Badge>
                <Badge colorScheme="green">Conf: {Math.round(confidence * 100)}%</Badge>
                <Badge colorScheme="blue">{strategy}</Badge>
            </HStack>

            <Box mt={3}>
                <StarRating rating={userRating} onChange={handleRating} />
            </Box>
        </Box>
    );
}
