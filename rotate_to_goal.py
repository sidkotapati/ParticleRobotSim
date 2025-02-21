import pymunk
import numpy as np
import pygame
import pymunk.pygame_util
import math

# Constants
radius = 0.05
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

# Create particles/robots
num_robots = 10
robots = []

# Start with a center particle
center_pos = (0, 0)
center_body = pymunk.Body()
center_shape = pymunk.Circle(center_body, radius)
center_shape.mass = 1.0
center_shape.friction = 0.9
center_shape.elasticity = 0.3
center_shape.color = (50, 50, 50, 255)
center_body.position = center_pos
center_body.velocity = (0, 0)
space.add(center_body, center_shape)
robots.append(center_body)

# Add robots in a ring around the center
for i in range(min(6, num_robots-1)):
    angle = 2 * np.pi * i / min(6, num_robots-1)
    pos_x = center_pos[0] + 2 * radius * np.cos(angle)
    pos_y = center_pos[1] + 2 * radius * np.sin(angle)
    
    body = pymunk.Body()
    shape = pymunk.Circle(body, radius)
    shape.mass = 1.0
    shape.friction = 0.9
    shape.elasticity = 0.3
    shape.color = (100, 100, 100, 255)
    
    body.position = (pos_x, pos_y)
    body.velocity = (0, 0)
    
    space.add(body, shape)
    robots.append(body)

# If we need more robots, add another layer
if num_robots > 7:
    current_count = len(robots)
    layer_radius = 2 * 2 * radius
    
    second_layer_count = min(12, num_robots - current_count)
    
    for i in range(second_layer_count):
        angle = 2 * np.pi * i / second_layer_count
        pos_x = center_pos[0] + layer_radius * np.cos(angle)
        pos_y = center_pos[1] + layer_radius * np.sin(angle)
        
        # Find closest robot to ensure touching
        closest_dist = float('inf')
        closest_pos = None
        
        for p in robots:
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
            
        final_pos = closest_pos + direction * 2 * radius
        
        body = pymunk.Body()
        shape = pymunk.Circle(body, radius)
        shape.mass = 10.0
        shape.friction = 0.9
        shape.elasticity = 0.3
        shape.color = (150, 150, 150, 255)
        
        body.position = tuple(final_pos)
        body.velocity = (0, 0)
            
        space.add(body, shape)
        robots.append(body)

# Function to draw rotation indicators
def draw_rotation_indicators():
    for body in robots:
        pos = body.position
        angle = body.angle
        screen_pos = (
            int((pos[0] + display_offset[0]) * display_scale),
            int((display_offset[1] - pos[1]) * display_scale)
        )
        end_x = screen_pos[0] + int(radius * display_scale * np.cos(angle))
        end_y = screen_pos[1] - int(radius * display_scale * np.sin(angle))
        pygame.draw.line(screen, (255, 0, 0), screen_pos, (end_x, end_y), 2)

# Goal point for the robots to reach
goal_point = (0.2, 0.2)

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

# Function to find the outer perimeter robots
def find_perimeter_robots(robots):
    # Find center of mass
    com_x = sum(r.position.x for r in robots) / len(robots)
    com_y = sum(r.position.y for r in robots) / len(robots)
    com = np.array([com_x, com_y])
    
    # Sort robots by distance from center of mass
    robots_by_dist = sorted(robots, 
                           key=lambda r: -np.linalg.norm(np.array(r.position) - com))
    
    # Return the 30% furthest robots as perimeter robots
    num_perimeter = max(3, int(len(robots) * 0.3))
    return robots_by_dist[:num_perimeter]

# Function to move a robot around the perimeter
def move_robot_around_perimeter(moving_robot, robots, goal_point):
    # Find center of mass excluding the moving robot
    other_robots = [r for r in robots if r != moving_robot]
    if not other_robots:
        return
    
    com_x = sum(r.position.x for r in other_robots) / len(other_robots)
    com_y = sum(r.position.y for r in other_robots) / len(other_robots)
    com = np.array([com_x, com_y])
    
    # Direction from COM to goal
    goal_dir = np.array(goal_point) - com
    if np.linalg.norm(goal_dir) > 0:
        goal_dir = goal_dir / np.linalg.norm(goal_dir)
    
    # Current position relative to COM
    rel_pos = np.array(moving_robot.position) - com
    
    # Calculate target position - use goal direction but maintain distance from center
    dist_from_center = np.linalg.norm(rel_pos)
    
    # Calculate angle to rotate around COM towards goal
    current_angle = math.atan2(rel_pos[1], rel_pos[0])
    goal_angle = math.atan2(goal_dir[1], goal_dir[0])
    
    # Determine rotation direction (clockwise or counterclockwise)
    angle_diff = (goal_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
    rotation_step = min(0.05, abs(angle_diff)) * np.sign(angle_diff)
    
    # Calculate new position
    new_angle = current_angle + rotation_step
    new_pos = com + dist_from_center * np.array([math.cos(new_angle), math.sin(new_angle)])
    
    # Apply force to move towards the new position
    force_dir = new_pos - np.array(moving_robot.position)
    force_magnitude = 5.0
    if np.linalg.norm(force_dir) > 0:
        force = force_dir / np.linalg.norm(force_dir) * force_magnitude
        moving_robot.apply_force_at_local_point(tuple(force), (0, 0))
    
    # Color the moving robot red to visualize it
    for shape in moving_robot.shapes:
        shape.color = (255, 0, 0, 255)

# Function to reset robot colors
def reset_robot_colors(robots):
    for i, robot in enumerate(robots):
        if i == 0:
            color = (50, 50, 50, 255)  # Center robot
        elif i < 7:
            color = (100, 100, 100, 255)  # First ring
        else:
            color = (150, 150, 150, 255)  # Outer ring
        
        for shape in robot.shapes:
            shape.color = color

# Function to draw the goal point
def draw_goal_point(goal_point):
    goal_screen_pos = (
        int((goal_point[0] + display_offset[0]) * display_scale),
        int((display_offset[1] - goal_point[1]) * display_scale)
    )
    pygame.draw.circle(screen, (0, 255, 0), goal_screen_pos, 10)
    pygame.draw.circle(screen, (0, 0, 0), goal_screen_pos, 10, 2)

# Function to check if any robot has reached the goal
def check_goal_reached(robots, goal_point, threshold=0.05):
    for robot in robots:
        dist = np.linalg.norm(np.array(robot.position) - np.array(goal_point))
        if dist < threshold:
            return True, robot
    return False, None

# Simulation loop
time = 0
time_steps = 100000
dt = 0.02
move_cooldown = 0
current_moving_robot = None
goal_reached = False
reached_robot = None

# Randomize goal point location periodically
next_goal_change = 20.0  # Change goal every 20 seconds
last_goal_change = 0

for step in range(time_steps):
    # Check if we need to change the goal point
    if time - last_goal_change > next_goal_change and not goal_reached:
        goal_point = (np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
        last_goal_change = time
        print(f"New goal point: {goal_point}")

    # Check if any robot has reached the goal
    goal_reached, reached_robot = check_goal_reached(robots, goal_point)
    
    if not goal_reached:
        # Find perimeter robots
        perimeter_robots = find_perimeter_robots(robots)
        
        # Select robot to move (furthest from goal among perimeter robots)
        if move_cooldown <= 0 or current_moving_robot is None:
            reset_robot_colors(robots)
            perimeter_dists = [(r, np.linalg.norm(np.array(r.position) - np.array(goal_point))) 
                              for r in perimeter_robots]
            perimeter_dists.sort(key=lambda x: -x[1])  # Sort by distance, furthest first
            
            if perimeter_dists:
                current_moving_robot = perimeter_dists[0][0]
                move_cooldown = 1.0  # Time to move this robot before selecting a new one
        
        # Move the selected robot around the perimeter
        if current_moving_robot:
            move_robot_around_perimeter(current_moving_robot, robots, goal_point)
            move_cooldown -= dt
    else:
        # Highlight the robot that reached the goal
        reset_robot_colors(robots)
        if reached_robot:
            for shape in reached_robot.shapes:
                shape.color = (0, 255, 0, 255)  # Green color for the robot that reached the goal
    
    # Apply attraction forces to keep the robots together
    for i, body1 in enumerate(robots):
        for j, body2 in enumerate(robots[i+1:], i+1):
            pos1 = np.array(body1.position)
            pos2 = np.array(body2.position)
            dx = pos2 - pos1
            distance = np.linalg.norm(dx)
            
            if distance < 2.5 * radius:  # Attraction range
                attract_strength = 2.0
                force_dir = dx / max(distance, 0.0001)
                attract_force = force_dir * attract_strength
                
                body1.apply_force_at_local_point(tuple(attract_force), (0, 0))
                body2.apply_force_at_local_point(tuple(-attract_force), (0, 0))
    
    # Add velocity damping
    for body in robots:
        vel = np.array(body.velocity)
        body.velocity = tuple(vel * 0.90)
    
    space.step(dt)
    time += dt

    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Allow user to set a new goal point by clicking
            mouse_pos = pygame.mouse.get_pos()
            # Convert screen coordinates to simulation coordinates
            goal_point = (
                mouse_pos[0]/display_scale - display_offset[0],
                display_offset[1] - mouse_pos[1]/display_scale
            )
            print(f"New goal point: {goal_point}")
            goal_reached = False
            reached_robot = None
    
    # Draw
    screen.fill((255, 255, 255))
    
    # Draw goal point first (so it's behind robots)
    draw_goal_point(goal_point)
    
    # Draw robots and space
    space.debug_draw(draw_options)
    draw_rotation_indicators()
    
    # Draw text info
    if goal_reached:
        font = pygame.font.SysFont(None, 36)
        text = font.render("Goal Reached!", True, (0, 150, 0))
        screen.blit(text, (50, 50))
        
        # Draw new goal instructions
        inst_font = pygame.font.SysFont(None, 24)
        inst_text = inst_font.render("Click anywhere to set a new goal", True, (100, 100, 100))
        screen.blit(inst_text, (50, 90))
    
    pygame.display.update()
    clock.tick(60)

    if step % 10 == 0:
        fps = int(clock.get_fps())
        print(f"{fps} fps / time = {time:.2f} / {'Goal Reached' if goal_reached else 'Moving...'}", end="\r")