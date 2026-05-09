Эксперимент2 по влиянию анализу важности позиции признаков.

Гипотеза:
Если позиция шипика на дендрите не имеет щанчения ,то при сильной аугментации или обмене позиции мы не потерям в точности.

Метод:
Обучаем энкодер но вносим измененя в признаки
(Перед этим надо проверить что признаки в графе приведены к относительным величинам)
Прибавляем к позиции шипика гаусс с разной дисперсией
sigma = [10 , 1000, 10000]

После чего тестируем резултаты классфикации на разных датасетах с такой же аугментацией позиции.


Сравнитть результаты классфикации с исзодными  и сделать выводы.

Здача сейчас:

Написать код в папке experiment_shuffle , который обучит ряд энкодеров на датасетах с аугментацией в позиции шипиков.
Важно что следует сохранять модель по аналогии с сохранение модели в папке liting_loss - то есть папка модели /version/checkpoints ,hparms , tensorboard метрики.

Сохранять модели по пути src/crash_tests/experiments_shufle



torch_5060) eugen@eugen-DRB-P:~/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor$ python -m src.crash_tests.experiments_shufle.evaluate_pos_noise
Seed set to 51
Found 3 encoder(s):
  pos_sigma=    10.0  →  src/crash_tests/experiments_shufle/jepa_pos_sigma_10/version_0
  pos_sigma=  1000.0  →  src/crash_tests/experiments_shufle/jepa_pos_sigma_1000/version_1
  pos_sigma= 10000.0  →  src/crash_tests/experiments_shufle/jepa_pos_sigma_10000/version_0

============================================================
  Position Noise σ = 10.0
  Encoder: src/crash_tests/experiments_shufle/jepa_pos_sigma_10/version_0
============================================================
Loading encoder from: src/crash_tests/experiments_shufle/jepa_pos_sigma_10/version_0
Computing dynamic macro statistics for dataset...
Embedding dimension: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores

Starting classifier training: pos_sigma_10.0
2026-05-02 16:05:30.980831: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-05-02 16:05:31.052209: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI AVX_VNNI_INT8 AVX_NE_CONVERT FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-05-02 16:05:32.483176: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 999/999 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 2.000 train_loss:  
                                                                      0.261 train_acc: 0.917    
Epoch 999/999 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 2.000 train_loss:  
                                                                      0.261 train_acc: 0.917    
                                                                      train_f1: 0.915 val_loss: 
                                                                      1.005 val_acc: 0.583      
                                                                      val_f1: 0.580             

Running evaluation on test set: pos_sigma_10.0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            1.0            │
│          test_f1          │            1.0            │
│         test_loss         │    0.10472070425748825    │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  
------------------------------------------------------------

============================================================
  Position Noise σ = 1000.0
  Encoder: src/crash_tests/experiments_shufle/jepa_pos_sigma_1000/version_1
============================================================
Loading encoder from: src/crash_tests/experiments_shufle/jepa_pos_sigma_1000/version_1
Computing dynamic macro statistics for dataset...
Embedding dimension: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores

Starting classifier training: pos_sigma_1000.0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 999/999 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.077 train_acc: 1.000    
Epoch 999/999 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.077 train_acc: 1.000    
                                                                      train_f1: 1.000 val_loss: 
                                                                      0.760 val_acc: 0.750      
                                                                      val_f1: 0.733             

Running evaluation on test set: pos_sigma_1000.0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            0.0            │
│          test_f1          │            0.0            │
│         test_loss         │    1.1779922246932983     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  
------------------------------------------------------------

============================================================
  Position Noise σ = 10000.0
  Encoder: src/crash_tests/experiments_shufle/jepa_pos_sigma_10000/version_0
============================================================
Loading encoder from: src/crash_tests/experiments_shufle/jepa_pos_sigma_10000/version_0
Computing dynamic macro statistics for dataset...
Embedding dimension: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores

Starting classifier training: pos_sigma_10000.0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 999/999 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.218 train_acc: 0.938    
Epoch 999/999 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.218 train_acc: 0.938    
                                                                      train_f1: 0.936 val_loss: 
                                                                      1.008 val_acc: 0.500      
                                                                      val_f1: 0.438             

Running evaluation on test set: pos_sigma_10000.0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            1.0            │
│          test_f1          │            1.0            │
│         test_loss         │    0.5309976935386658     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  
------------------------------------------------------------

(torch_5060) eugen@eugen-DRB-P:~/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor$  


Результаты:

Сравнительно слабые резульатты это можно инетпретировать как то что дял данной арзитектуры взаимное располоение шипиков играет важною роль.



Также есть результаты нормально обученной модели но с шумом уже на этапе инференса


(torch_5060) eugen@eugen-DRB-P:~/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor$ python -m src.crash_tests.experiments_shufle.evaluate_noise_robustness
Seed set to 51
============================================================
  Robustness Test: Position Noise Sweep
  Encoder:  /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0
  Sigmas:   [0, 10, 100, 1000, 10000]
============================================================

============================================================
  pos_noise_sigma = 0
============================================================
  Loading encoder from: /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0
  Computing dynamic macro statistics for dataset...
  Embedding dim: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
  Training classifier: noise_0
2026-05-02 16:32:49.891922: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-05-02 16:32:49.962089: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI AVX_VNNI_INT8 AVX_NE_CONVERT FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-05-02 16:32:51.371366: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 1.000 train_loss:  
                                                                      0.393 train_acc: 0.917    
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 1.000 train_loss:  
                                                                      0.393 train_acc: 0.917    
                                                                      train_f1: 0.913 val_loss: 
                                                                      0.452 val_acc: 0.917      
                                                                      val_f1: 0.911             
  Evaluating on test set: noise_0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            1.0            │
│          test_f1          │            1.0            │
│         test_loss         │    0.3756919503211975     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  

  ✓ σ=0  acc=1.0000  f1=1.0000  loss=0.3757
------------------------------------------------------------

============================================================
  pos_noise_sigma = 10
============================================================
  Loading encoder from: /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0
  Computing dynamic macro statistics for dataset...
  Embedding dim: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
  Training classifier: noise_10
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.422 train_acc: 0.792    
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.422 train_acc: 0.792    
                                                                      train_f1: 0.786 val_loss: 
                                                                      0.619 val_acc: 0.667      
                                                                      val_f1: 0.657             
  Evaluating on test set: noise_10
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            0.0            │
│          test_f1          │            0.0            │
│         test_loss         │    1.0162664651870728     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  

  ✓ σ=10  acc=0.0000  f1=0.0000  loss=1.0163
------------------------------------------------------------

============================================================
  pos_noise_sigma = 100
============================================================
  Loading encoder from: /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0
  Computing dynamic macro statistics for dataset...
  Embedding dim: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
  Training classifier: noise_100
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.364 train_acc: 0.896    
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.364 train_acc: 0.896    
                                                                      train_f1: 0.894 val_loss: 
                                                                      0.512 val_acc: 0.750      
                                                                      val_f1: 0.733             
  Evaluating on test set: noise_100
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            1.0            │
│          test_f1          │            1.0            │
│         test_loss         │    0.34013229608535767    │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  

  ✓ σ=100  acc=1.0000  f1=1.0000  loss=0.3401
------------------------------------------------------------

============================================================
  pos_noise_sigma = 1000
============================================================
  Loading encoder from: /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0
  Computing dynamic macro statistics for dataset...
  Embedding dim: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
  Training classifier: noise_1000
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.378 train_acc: 0.854    
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.378 train_acc: 0.854    
                                                                      train_f1: 0.849 val_loss: 
                                                                      0.569 val_acc: 0.667      
                                                                      val_f1: 0.667             
  Evaluating on test set: noise_1000
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            1.0            │
│          test_f1          │            1.0            │
│         test_loss         │    0.2831050157546997     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  

  ✓ σ=1000  acc=1.0000  f1=1.0000  loss=0.2831
------------------------------------------------------------

============================================================
  pos_noise_sigma = 10000
============================================================
  Loading encoder from: /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0
  Computing dynamic macro statistics for dataset...
  Embedding dim: 39 (Encoder: 32, Macro: 7)
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'encoder_graph' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['encoder_graph'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
  Training classifier: noise_10000
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ encoder_graph │ GraphLatent      │  9.8 K │ train │     0 │
│ 1 │ classifier    │ LinearClassifier │    422 │ train │     0 │
│ 2 │ loss_fn       │ CrossEntropyLoss │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 422                                                                           
Non-trainable params: 9.8 K                                                                     
Total params: 10.2 K                                                                            
Total estimated model params size (MB): 0                                                       
Modules in train mode: 8                                                                        
Modules in eval mode: 27                                                                        
Total FLOPs: 0                                                                                  
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:365: Skipping 'encoder_graph' parameter because it is not possible to safely dump to YAML.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/
_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, 
TreeSpec) and treespec.is_leaf()` instead.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:317: The number of training batches (1) is smaller than the logging interval 
Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs 
for the training epoch.
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/loops/fit_
loop.py:534: Found 27 module(s) in eval mode at the start of training. This may lead to 
unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.406 train_acc: 0.875    
Epoch 149/149 ━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s v_num: 0.000 train_loss:  
                                                                      0.406 train_acc: 0.875    
                                                                      train_f1: 0.869 val_loss: 
                                                                      0.542 val_acc: 0.833      
                                                                      val_f1: 0.829             
  Evaluating on test set: noise_10000
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/eugen/miniforge3/envs/torch_5060/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │            1.0            │
│          test_f1          │            1.0            │
│         test_loss         │    0.5925321578979492     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s  

  ✓ σ=10000  acc=1.0000  f1=1.0000  loss=0.5925
------------------------------------------------------------


============================================================
  ROBUSTNESS SUMMARY
  Encoder: /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0
============================================================
       sigma    test_acc     test_f1   test_loss
  ──────────  ──────────  ──────────  ──────────
           0      1.0000      1.0000      0.3757
          10      0.0000      0.0000      1.0163
         100      1.0000      1.0000      0.3401
        1000      1.0000      1.0000      0.2831
       10000      1.0000      1.0000      0.5925
============================================================
(torch_5060) eugen@eugen-DRB-P:~/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor$  

Мы видим что эти резултаты лучше чем в первом жксперименте однако остувиее шума показывает наилучшие резульат.