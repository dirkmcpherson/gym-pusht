import gymnasium as gym
import gym_pusht
import cv2
# add a path
import sys
sys.path.append("/home/james/workspace/fastrl")
# from nov20.second_wind.peripheral import get_memorymaze_action_from_joystick, get_pusht_action_from_joystick, get_calvin_action_from_joystick, get_pinpad_action_from_joystick, get_ev3_action_from_joystick

env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_state", render_mode="rgb_array", max_episode_steps=5000)
observation, info = env.reset()
import os
import pygame
pygame.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = "1"
import time
# for _ in range(1000):
while True:
    action = env.action_space.sample()
    # action = get_pusht_action_from_joystick(joysticks)
    # scale it up from -1, 1 to 0, 255
    # print(action, end=" --> ")
    # action = [(a + 1) * 255 for a in action]
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    print(observation.keys())
    cv2.imshow('obs', observation.transpose(2,0,1))

    cv2.imshow('img', image)
    cv2.waitKey(1)

    time.sleep(0.02)

    if terminated or truncated:
        observation, info = env.reset()

    #allow escape to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

env.close()
