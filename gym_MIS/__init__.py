from gym.envs.registration import register

register(
    id='MIS-v0',
    entry_point='gym_MIS.envs:MISEnv',
)