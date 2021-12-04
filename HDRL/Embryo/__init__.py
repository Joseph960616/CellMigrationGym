from gym.envs.registration import register

# Embryo env registeration
register(id='Embryo-v0',
         entry_point='Embryo.env:EmbryoBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)