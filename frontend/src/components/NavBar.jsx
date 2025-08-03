import React from 'react';
import { Box, VStack, IconButton, Tooltip, Flex } from '@chakra-ui/react';
import { NavLink } from 'react-router-dom';
import { Home, User, BarChart2 } from 'lucide-react';

// Navigation items
const navItems = [
    { label: 'Home', icon: Home, path: '/' },
    { label: 'Profile', icon: User, path: '/profile' },
    { label: 'Stats', icon: BarChart2, path: '/stats' },
];

export default function NavBar() {
    return (
        <Box
            as="nav"
            pos="fixed"
            top={0}
            left={0}
            h="100vh"
            w="80px"
            bgGradient="linear(to-b, #0B0C2D, #1E0F44)"
            boxShadow="xl"
            zIndex={1000}
        >
            <Flex direction="column" align="center" mt={8}>
                <VStack spacing={6}>
                    {navItems.map(({ label, icon: Icon, path }) => (
                        <NavLink
                            key={label}
                            to={path}
                            style={({ isActive }) => ({
                                width: '100%',
                                textAlign: 'center',
                                borderLeft: isActive ? '4px solid #7F5AF0' : '4px solid transparent',
                            })}
                        >
                            <Tooltip label={label} placement="right" openDelay={300}>
                                <IconButton
                                    aria-label={label}
                                    icon={<Icon size={24} />}
                                    variant="ghost"
                                    color="whiteAlpha.600"
                                    _hover={{ color: 'white', bg: 'transparent' }}
                                    _active={{ color: 'white', bg: 'transparent' }}
                                />
                            </Tooltip>
                        </NavLink>
                    ))}
                </VStack>
            </Flex>
        </Box>
    );
}
