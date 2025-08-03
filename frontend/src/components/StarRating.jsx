import React from 'react';
import { HStack, Icon } from '@chakra-ui/react';
import { Star } from 'lucide-react';

/**
 * Props:
 * - rating: number (0–5)
 * - onChange: (newRating: number) => void
 */
export default function StarRating({ rating = 0, onChange }) {
    return (
        <HStack spacing={1}>
            {[1, 2, 3, 4, 5].map((i) => (
                <Icon
                    key={i}
                    as={Star}
                    boxSize={5}
                    cursor="pointer"
                    color={i <= rating ? 'yellow.400' : 'gray.600'}
                    _hover={{ color: 'yellow.300' }}
                    onClick={() => onChange(i)}
                    aria-label={`${i} star${i > 1 ? 's' : ''}`}
                />
            ))}
        </HStack>
    );
}
