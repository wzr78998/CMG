This is a code demo for the paper “Inducing Causal Meta-Knowledge from Virtual Domains: Causal Meta-Generalization for Hyperspectral Domain Generalization”. 
1. This project provides a code demo for the Pavia dataset. You can run our code by executing CMG.py. Considering the randomness in different environments, we provide a set of model parameters. You can directly load the model parameters by setting train_begin and vae_begin to False in CMG.py. If you want to train from scratch, you can set these two parameters to True.The data can be placed under the CMG/data/Pavia directory.
2. The version of Torch we used is 2.1.2+cu121. Using an incorrect version of Torch may result in errors.
3. Reminder: If you encounter an 'out of memory' error, you can set use_sub=True. Using fewer samples may lead to some performance degradation, but it can reduce the memory overhead of the code.
4. The code defaults to loading the parameters we provide. If you want to train the meta-learning part from scratch, you can set: train_begin=False; vae_begin=False; use_parameters=True. If you want to train the entire code from scratch, you can set: train_begin=True; vae_begin=True; use_parameters=False.
5. If you have used our code, please cite our paper: H. Wang, X. Liu, Z. Qiao, and H. Tao, 'Inducing Causal Meta-knowledge from Virtual Domain: Causal Meta-generalization for Hyperspectral Domain Generalization,' in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3494796.

