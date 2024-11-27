The code is in the master branch of the repository. You can click the small arrow next to "main" in the upper left corner to view it.
This is a code demo for the paper “Inducing Causal Meta-Knowledge from Virtual Domains: Causal Meta-Generalization for Hyperspectral Domain Generalization”. 
1. This project provides a code demo for the Pavia and Houston dataset. You can run our code by executing CMG.py.
2. Considering the randomness in different environments, we provide a set of model parameters. The code defaults to loading the parameters we provide，we set: train_begin=False; vae_begin=False; use_parameters=True. If you want to train the meta-learning part from scratch, you can set: train_begin=False; vae_begin=False; use_parameters=False. If you want to train the entire code from scratch, you can set: train_begin=True; vae_begin=True; use_parameters=False.
3.The data can be placed under the CMG/data/Pavia directory.
4. The version of Torch we used is 2.1.2+cu121. Using an incorrect version of Torch may result in errors.
5. Reminder: If you encounter an 'out of memory' error, you can set use_sub=True. Using fewer samples may lead to some performance degradation, but it can reduce the memory overhead of the code.
6. If you have used our code, please cite our paper: H. Wang, X. Liu, Z. Qiao, and H. Tao, 'Inducing Causal Meta-knowledge from Virtual Domain: Causal Meta-generalization for Hyperspectral Domain Generalization,' in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3494796.

