import pymunk
import numpy as np
import pygame
import pymunk.pygame_util

# Constants
radius = 0.05
boundary_radius = 0.35

# Display setup
display_window = (800, 600)
display_scale = 800
pygame.init()
screen = pygame.display.set_mode(display_window)
clock = pygame.time.Clock()

# Pymunk display options
pymunk.pygame_util.positive_y_is_up = True
display_offset = np.array(display_window) / (2 * display_scale)
draw_options = pymunk.pygame_util.DrawOptions(screen)
draw_options.transform = pymunk.Transform.scaling(display_scale).translated(*display_offset)
draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

# Create space
space = pymunk.Space()
space.gravity = (0, 0)

# Create boundary
boundary_body = pymunk.Body(body_type=pymunk.Body.STATIC)
space.add(boundary_body)

boundary_points = []
num_segments = 32
for i in range(num_segments):
    angle = 2 * np.pi * i / num_segments
    x = boundary_radius * np.cos(angle)
    y = boundary_radius * np.sin(angle)
    boundary_points.append((x, y))

# Create boundary segments
for i in range(num_segments):
    p1 = boundary_points[i]
    p2 = boundary_points[(i + 1) % num_segments]
    segment = pymunk.Segment(boundary_body, p1, p2, 0.01)
    segment.friction = 1.0
    segment.elasticity = 0.9
    space.add(segment)

# Create particles
# Create particles in contact with each other
num_particles = 25
particles = []

# Start with a center particle
center_pos = (0, 0)
center_body = pymunk.Body()
center_shape = pymunk.Circle(center_body, radius)
center_shape.mass = 1000000.0
center_shape.friction = 1000
center_shape.elasticity = 0.9
center_shape.color = (128, 128, 128, 255)
center_body.position = center_pos
center_body.velocity = (0, 0)  # Start at rest
space.add(center_body, center_shape)
particles.append(center_body)

# Add particles in a ring around the center
for i in range(min(6, num_particles-1)):  # Up to 6 in first ring
    angle = 2 * np.pi * i / min(6, num_particles-1)
    # Position exactly touching the center particle (2*radius apart from center to center)
    pos_x = center_pos[0] + 2 * radius * np.cos(angle)
    pos_y = center_pos[1] + 2 * radius * np.sin(angle)
    
    body = pymunk.Body()
    shape = pymunk.Circle(body, radius)
    shape.mass = 100000
    shape.friction = 0.7
    shape.elasticity = 0.9
    shape.color = (128, 128, 128, 255)
    
    body.position = (pos_x, pos_y)
    body.velocity = (0, 0)  # Start at rest
    
    space.add(body, shape)
    particles.append(body)

# If we need more particles, add another layer
if num_particles > 7:
    current_count = len(particles)
    layer_radius = 2 * 2 * radius  # Distance from center to second layer
    
    # Adjust number in second layer based on remaining particles needed
    second_layer_count = min(12, num_particles - current_count)
    
    for i in range(second_layer_count):
        angle = 2 * np.pi * i / second_layer_count
        # Position particles in second ring
        pos_x = center_pos[0] + layer_radius * np.cos(angle)
        pos_y = center_pos[1] + layer_radius * np.sin(angle)
        
        # Adjust position slightly to ensure touching
        # Find closest particle and move towards it
        closest_dist = float('inf')
        closest_pos = None
        
        for p in particles:
            p_pos = np.array(p.position)
            new_pos = np.array((pos_x, pos_y))
            dist = np.linalg.norm(p_pos - new_pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_pos = p_pos
        
        direction = np.array((pos_x, pos_y)) - closest_pos
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array((1, 0))  # Default direction if same position
            
        # Position exactly touching the closest particle
        final_pos = closest_pos + direction * 2 * radius
        
        body = pymunk.Body()
        shape = pymunk.Circle(body, radius)
        shape.mass = 1.0
        shape.friction = 0.7
        shape.elasticity = 0.9
        shape.color = (128, 128, 128, 255)
        
        body.position = tuple(final_pos)
        body.velocity = (0, 0)  # Start at rest
        
        space.add(body, shape)
        particles.append(body)

# Simulation loop
time = 0
time_steps = 10000
dt = 0.02

for step in range(time_steps):
    for i, body1 in enumerate(particles):
        for body2 in particles[i+1:]:
            # Get displacement vector between particles
            pos1 = np.array(body1.position)
            pos2 = np.array(body2.position)
            dx = pos2 - pos1
            distance = np.linalg.norm(dx)
            
            if distance < 0.100001: # Interaction radius
                # Calculate orbital force
                tangent = np.array([dx[1], -dx[0]]) / distance
                orbital_strength = 5
                
                # Apply tangential forces to create orbital motion
                force = tangent * orbital_strength
                print(force)
                body1.apply_force_at_local_point(tuple(-force), (0, 0))
                body2.apply_force_at_local_point(tuple(force), (0, 0))
                
                # Add attractive force to keep particles from drifting apart
                attract_strength = 2.0
                force_dir = dx / distance
                attract_force = force_dir * attract_strength
                body1.apply_force_at_local_point(tuple(attract_force), (0, 0))
                body2.apply_force_at_local_point(tuple(-attract_force), (0, 0))
    
        # Add damping
        vel = np.array(body1.velocity)
        body1.velocity = tuple(vel * 0.90)
    
    space.step(dt)
    time += dt

    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    # Draw
    screen.fill((255, 255, 255))
    space.debug_draw(draw_options)
    pygame.display.update()
    clock.tick(60)

    print(f"{int(clock.get_fps())} fps / time = {time:.2f}", end="\r")