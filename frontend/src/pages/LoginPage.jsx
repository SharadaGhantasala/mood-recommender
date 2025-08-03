import React, { useState } from 'react';
import {
    Box,
    Heading,
    VStack,
    Input,
    Button,
    Text,
    Link,
    FormControl,
    FormLabel,
    FormHelperText,
    useColorModeValue
} from '@chakra-ui/react';

export default function LoginPage({ onAuthModeChange }) {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const bg = useColorModeValue('whiteAlpha.100', 'whiteAlpha.200');

    const handleSubmit = (e) => {
        e.preventDefault();
        // placeholder: call login or register API
        console.log('authenticate', { email, password });
    };

    return (
        <Box
            maxW="md"
            mx="auto"
            mt={20}
            p={8}
            bg={bg}
            borderRadius="lg"
            boxShadow="2xl"
        >
            <Heading mb={6} textAlign="center">
                Welcome Back
            </Heading>

            <form onSubmit={handleSubmit}>
                <VStack spacing={4}>
                    <FormControl id="email">
                        <FormLabel>Email address</FormLabel>
                        <Input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="you@example.com"
                            required
                        />
                        <FormHelperText>We'll never share your email.</FormHelperText>
                    </FormControl>

                    <FormControl id="password">
                        <FormLabel>Password</FormLabel>
                        <Input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="••••••••"
                            required
                        />
                    </FormControl>

                    <Button type="submit" colorScheme="purple" w="full">
                        Log In
                    </Button>

                    <Text fontSize="sm">
                        First time here?{' '}
                        <Link
                            color="teal.200"
                            onClick={() => onAuthModeChange('register')}
                        >
                            Create an account
                        </Link>
                    </Text>
                </VStack>
            </form>
        </Box>
    );
}
