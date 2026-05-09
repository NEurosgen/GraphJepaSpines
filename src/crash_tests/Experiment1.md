Краш тест обучение модели , при отсуствии фич.


Тест проверяет насколько модель опиарется на структурные признаки а не на морфологию шипиков при классицкацци данных

Датасет. Обучаемся на миниие 65 с графами

Далее адаптируемся к задаче классификации на альцгеймере.

Метод:
Проодим в графе ребра а потом примениям гаусс к признакам узлов с N(0, sigma)
Так напрмиер обучаем  три модели с разными аугментациями
делаем сигма 10 , 100, 10000

После чего смотрим их на задаче классифкации


Описание задачи:
В файле empty_nodes.py приведен фрагмент кода для другой задачи. Его следует адаптировать под следующую задачу.

Код должен создавать датасет по конфигу из config/config.yaml далее ко всем элементам даатсета применяется аугментация и начинается полседовательное обучение модели.
Важно что все обученный модели созхраняются в папку crash_tests/empty_nodes/



Результаты:
torch_5060) eugen@eugen-DRB-P:~/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor$ python -m src.crash_tests.empty_nodes
============================================================
  Gaussian Noise Crash Test
  Sigma values: [10, 100, 10000]
============================================================
  pos_std = torch.std(pos,dim=0).clamp(min=1e-6)
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type   ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ LeJEPA │ 18.5 K │ train │     0 │
└───┴───────┴────────┴────────┴───────┴───────┘
Trainable params: 15.3 K                                                                                                               
Non-trainable params: 3.2 K                                                                                                            
Total params: 18.5 K                                                                                                                   
Total estimated model params size (MB): 0                                                                                              
Modules in train mode: 40                                                                                                              
Modules in eval mode: 0                                                                                                                
Total FLOPs: 0                                                                                                                         
batch_size=batch_size)`.
Epoch 199/199 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57/57 0:00:22 • 0:00:00 2.53it/s v_num: 0.000 train_loss: 0.785 val_loss: 0.854`Trainer.fit` stopped: `max_epochs=200` reached.
Epoch 199/199 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57/57 0:00:22 • 0:00:00 2.53it/s v_num: 0.000 train_loss: 0.785 val_loss: 0.859
Finished Training for sigma = 10

Clearing RAM and GPU memory for sigma = 10...

==================================================
   Starting iteration for sigma = 100
==================================================
Removing old prepared dataset /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/datasets/dataset_prepared...
Preparing dataset with sigma=100...
Done preparing. Initializing Model...                                                                                                  
Seed set to 51
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
Starting Training for sigma = 100
[2026-05-01 23:40:32,515][root][INFO] - Pre-loading dataset into RAM cache...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type   ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ LeJEPA │ 18.5 K │ train │     0 │
└───┴───────┴────────┴────────┴───────┴───────┘
Trainable params: 15.3 K                                                                                                               
Non-trainable params: 3.2 K                                                                                                            
Total params: 18.5 K                                                                                                                   
Total estimated model params size (MB): 0                                                                                              
Modules in train mode: 40                                                                                                              
Modules in eval mode: 0                                                                                                                
Total FLOPs: 0                                                                                                                         
Epoch 199/199 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57/57 0:00:12 • 0:00:00 4.80it/s v_num: 0.000 train_loss: 0.843 val_loss: 0.928`Trainer.fit` stopped: `max_epochs=200` reached.
Epoch 199/199 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57/57 0:00:12 • 0:00:00 4.80it/s v_num: 0.000 train_loss: 0.843 val_loss: 0.923
Finished Training for sigma = 100

Clearing RAM and GPU memory for sigma = 100...

==================================================
   Starting iteration for sigma = 10000
==================================================
Removing old prepared dataset /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/datasets/dataset_prepared...
Preparing dataset with sigma=10000...
Done preparing. Initializing Model...                                                                                                  
Seed set to 51
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
Starting Training for sigma = 10000
[2026-05-02 01:07:24,204][root][INFO] - Pre-loading dataset into RAM cache...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type   ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ LeJEPA │ 18.5 K │ train │     0 │
└───┴───────┴────────┴────────┴───────┴───────┘
Trainable params: 15.3 K                                                                                                               
Non-trainable params: 3.2 K                                                                                                            
Total params: 18.5 K                                                                                                                   
Total estimated model params size (MB): 0                                                                                              
Modules in train mode: 40                                                                                                              
Modules in eval mode: 0                                                                                                                
Total FLOPs: 0                                                                                                                         
Epoch 199/199 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57/57 0:00:16 • 0:00:00 3.49it/s v_num: 0.000 train_loss: 0.854 val_loss: 0.981`Trainer.fit` stopped: `max_epochs=200` reached.
Epoch 199/199 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57/57 0:00:16 • 0:00:00 3.49it/s v_num: 0.000 train_loss: 0.854 val_loss: 0.968
Finished Training for sigma = 10000

Clearing RAM and GPU memory for sigma = 10000...
(torch_5060) eugen@eugen-DRB-P:~/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor$ 


Выводы эксперимента:
После добавления шума к прищнакам модеь разделяет классы с  точность acc = 0.75 , f1 = 0.697