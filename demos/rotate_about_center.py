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
num_particles = 14
particles = []
inner_particles = []  # Track inner particles (center + first ring)
outer_particles = []  # Track outer ring particles

# Start with a center particle
center_pos = (0, 0)
center_body = pymunk.Body()
center_shape = pymunk.Circle(center_body, radius)
center_shape.mass = 1000000.0
center_shape.friction = 0.9
center_shape.elasticity = 0.3
center_shape.color = (50, 50, 50, 255)  # Darker gray for center
center_body.position = center_pos
center_body.velocity = (0, 0)
center_body.angular_velocity = 0.0  # No rotation for center
space.add(center_body, center_shape)
particles.append(center_body)
inner_particles.append(center_body)

# Add particles in a ring around the center (first ring - static)
for i in range(min(6, num_particles-1)):
    angle = 2 * np.pi * i / min(6, num_particles-1)
    pos_x = center_pos[0] + 2 * radius * np.cos(angle)
    pos_y = center_pos[1] + 2 * radius * np.sin(angle)
    
    body = pymunk.Body()
    shape = pymunk.Circle(body, radius)
    shape.mass = 100000
    shape.friction = 0.9
    shape.elasticity = 0.3
    
    # Medium gray for first ring
    shape.color = (100, 100, 100, 255)
    
    body.position = (pos_x, pos_y)
    body.velocity = (0, 0)
    body.angular_velocity = 0.0  # No rotation for first ring
    
    space.add(body, shape)
    particles.append(body)
    inner_particles.append(body)

# If we need more particles, add another layer (outer ring - rotating)
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
            direction = np.array((1, 0))
            
        # Position exactly touching the closest particle
        final_pos = closest_pos + direction * 2 * radius
        
        body = pymunk.Body()
        shape = pymunk.Circle(body, radius)
        shape.mass = 1.0
        shape.friction = 0.9
        shape.elasticity = 0.3
        
        # Light gray for outer ring (rotating particles)
        shape.color = (180, 180, 180, 255)
        
        body.position = tuple(final_pos)
        body.velocity = (0, 0)
        
        # Set initial rotation for outer ring
        direction = 1
        body.angular_velocity = 2.0 * direction
            
        space.add(body, shape)
        particles.append(body)
        outer_particles.append(body)

# Function to draw rotation indicators
def draw_rotation_indicators():
    for body in particles:
        pos = body.position
        angle = body.angle
        # Convert to screen coordinates
        screen_pos = (
            int((pos[0] + display_offset[0]) * display_scale),
            int((display_offset[1] - pos[1]) * display_scale)
        )
        # Draw line indicating rotation angle
        end_x = screen_pos[0] + int(radius * display_scale * np.cos(angle))
        end_y = screen_pos[1] - int(radius * display_scale * np.sin(angle))
        pygame.draw.line(screen, (255, 0, 0), screen_pos, (end_x, end_y), 2)

# Simulation loop
time = 0
time_steps = 10000
dt = 0.02

for step in range(time_steps):
    # Keep inner particles static
    for body in inner_particles:
        body.angular_velocity = 0.0
        
    # Maintain rotation for outer particles
    for i, body in enumerate(outer_particles):
        # Alternate direction based on position
        target_angular_vel = 15.0
            
        # Gradually adjust to target angular velocity
        current_vel = body.angular_velocity
        body.angular_velocity = current_vel * 0.95 + target_angular_vel * 0.05
    
    # Handle particle interactions
    for i, body1 in enumerate(particles):
        for j, body2 in enumerate(particles[i+1:], i+1):
            # Get displacement vector between particles
            pos1 = np.array(body1.position)
            pos2 = np.array(body2.position)
            dx = pos2 - pos1
            distance = np.linalg.norm(dx)
            
            if distance < 2.1 * radius:  # Slightly more than touching
                # Calculate attract force to keep particles from drifting apart
                attract_strength = 10.0
                force_dir = dx / max(distance, 0.0001)  # Avoid division by zero
                attract_force = force_dir * attract_strength
                
                # Apply attraction force
                body1.apply_force_at_local_point(tuple(attract_force), (0, 0))
                body2.apply_force_at_local_point(tuple(-attract_force), (0, 0))
    
    # Add velocity damping to prevent particles from flying apart
    for body in particles:
        vel = np.array(body.velocity)
        body.velocity = tuple(vel * 0.90)  # Damping factor
    
    # Force inner particles to maintain zero angular velocity
    for body in inner_particles:
        body.angular_velocity = 0.0
    
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
    draw_rotation_indicators()  # Draw rotation indicators
    pygame.display.update()
    clock.tick(60)

    if step % 10 == 0:  # Reduce frequency of FPS display for better performance
        print(f"{int(clock.get_fps())} fps / time = {time:.2f}", end="\r")