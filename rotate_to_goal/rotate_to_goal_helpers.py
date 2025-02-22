import pymunk
import numpy as np
import pygame
import pymunk.pygame_util
import time

# Function to find the robot furthest from the goal
def find_furthest_robot(robots, goal_point):
    max_dist = -1
    furthest_robot = None
    
    for robot in robots:
        dist = np.linalg.norm(np.array(robot.position) - np.array(goal_point))
        if dist > max_dist:
            max_dist = dist
            furthest_robot = robot
            
    return furthest_robot

def rotate_until_closest(curr_bot, particles, goal_point, space, radius=0.05):
    min_dist = float('inf')

    for particle in particles:
        dist = np.linalg.norm(np.array(particle.position) - np.array(goal_point))
        if dist < min_dist:
            min_dist = dist
    
    curr_dist = np.linalg.norm(np.array(curr_bot.position) - np.array(goal_point))
    while curr_dist > min_dist:
        # Handle particle interactions
        for i, body1 in enumerate(particles):
            for j, body2 in enumerate(particles[i+1:], i+1):
                # Get displacement vector between particles
                pos1 = np.array(body1.position)
                pos2 = np.array(body2.position)
                displacement = pos2 - pos1
                distance = np.linalg.norm(displacement)
                
                if distance < 2.1 * radius:  # Slightly more than touching
                    # Calculate attract force to keep particles from drifting apart
                    attract_strength = 3.0
                    force_dir = displacement / max(distance, 0.0001)  # Avoid division by zero
                    attract_force = force_dir * attract_strength
                    
                    # Apply attraction force
                    body1.apply_force_at_local_point(tuple(attract_force), (0, 0))
                    body2.apply_force_at_local_point(tuple(-attract_force), (0, 0))
        
        # Add velocity damping to prevent particles from flying apart
        for body in particles:
            vel = np.array(body.velocity)
            body.velocity = tuple(vel * 0.85)  # Damping factor

        space.step(0.02)
        #time.sleep(0.1)
        curr_dist = np.linalg.norm(np.array(curr_bot.position) - np.array(goal_point))
        print("curr_dist", curr_dist)

def check_goal_reached(robots, goal_point, threshold=0.05):
    for robot in robots:
        dist = np.linalg.norm(np.array(robot.position) - np.array(goal_point))
        if dist < threshold:
            return True, robot
    return False, None


