# DDPG_on_DMControl_Suite
Solving ceetah,cartpole,reacher,walker Deepmind Control Suite using DDPG

This repo is from my Master's degree thesis work.
I used PlaNet to prove that model-based DRL can overcome the model-free algorithms in terms of sample efficiency.
I extended the [MoritzTaylor](https://github.com/MoritzTaylor/ddpg-pytorch) implementation to make it compatible with the Deepmind Control Suite. 

To have more details on the full work, visit [blog article.](https://drlux.github.io/planpix.html)

# Results
------------
![ceetah_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/ceetah_planet_vs_ddpg.jpg?token=ADY2SMLWUAPFPPU4JNXKQS27YJYGS)
![cartpole_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/cartpole_planet_vs_ddpg.jpg?token=ADY2SMPTHHRUXMDSSM6UBR27YJYCQ)
![reacher_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/reacher_planet_vs_ddpg.jpg?token=ADY2SMM4XWCNZGRRW3JZWJC7YJYHU)
![walker_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/walker_planet_vs_ddpg.jpg?token=ADY2SMNTFGC7XWTOZY7XZTK7YJYJ2)


# Links
----------------
[My work with PlaNet]https://github.com/DrLux/Planpix
[MoritzTaylor implementation ]https://github.com/MoritzTaylor/ddpg-pytorch
[OpenAi implementation]https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
