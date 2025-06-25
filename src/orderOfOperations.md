### Main.py
1. Paramaters are taken in by commandLine
2. Timer is started
   3. Train and eval function is called with commands
      ### train.py
       1. Log opened data added
      2. Cuda or cpu is selected 
      3. DamageDataset object created 
           ### dataset.py
         1. For every post disaster GT file:
            - For every patch:
              - include if it contains class 2,3,4 or 10% of other classes
              - For those included add to sample list with coordinates of patch and name of file; record if it is of classes [2,3,4]
      ### train.py
      4. Print distribution of data
      5. Split data between Training and Validation 80/20
      6. Load training and validation data onto GPU
         7. Load EnhancedDamageModel
            ### model.py
                1. Loading resnet50 pretrained as the backbone with 3 trainable layers
                2. 