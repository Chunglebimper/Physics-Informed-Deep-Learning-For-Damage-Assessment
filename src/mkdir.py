import os

"""
Features:
 * Detect new run in and put folder in directory /results

 Process:
  * Identify where paths must be changed
    - log file
    - visual.py
  * read in folder names inside /results
  * read run number 
  * create new folder with higher run number
  * update all new uses of folder location
"""


def mkdir_results():
    os.makedirs('../results', exist_ok=True)  # create output directory
    highest = -1
    for folder in os.listdir('../results',):  # inside directory
        if int(folder) > highest:
            highest = int(folder)
    highest += 1
    return os.path.join('../results', f'{highest}')

