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
          2. FILL ME IN 
   ### train.py
     8. call get class weights for training set
          ### utils.py (mask images)
        1. For all collected samples in dataset[idx], append the dataset[idx] flattened image array to a list _BEWARE INDEX USES THE DATA AUGMENTATION BEHAVIOR OF DATASET see dataset.py_ ___getitem__ _
        2. Weights are auto balanced with sklearn given the number of classes and where y = 'Array of original class labels per sample.'
        3. class_weight_dict created
        4. Apply manual scaling to class_weight_dict to focus on rare damage classes
        5. Print weights and send them to tensor
    ### train.py
   9. Initialze object for loss function with weights for each class and initilize optimizer
   10. Additional initializing for best epochs, graphing, and general history
   11. Begin epoch loop for epoch in epochs
       12. start epoch timer
       13. Set the EnhancedDamageModel to training mode (model.py inherits from nn.Module to train)
       14. Freeze the EnhancedDamageModel backbone layers if epoch is <=4
       15. loss initialized
       16. for pre, post, mask arrays in train_loader 
           17. send them to the GPU
           18. reset gradients (optimizer)
           19. damage_out recorded from forward pass of EnhancedDamageModel
           20. loss_ce calculated using loss_fn 
           21. TO BE FILLED COUNTINUED