
# Speaker recognition using X-Vector feature extraction and SVM for classification 
This project was made for diploma work.

## Requirements
- Clone repo: `gh repo clone ordogalen/Szakdolgozat`
- Install CONDA and PyTorch on your device: 
	- cuda: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
	- PyTorch: https://pytorch.org/get-started/locally/
	
## Running application
- For training x-vector: (dataset.py)
	- Run`create_directories` method
	- Add your path to audio_files argument and start x_vector training
- Feature extracting: (feature_extractor.py)
	- After training is done, add your path to the variables and `run extract_features_into_file` method
	- Add your path-s to variable in SVM.py and run it.

#### Reference
- This project was based on: https://github.com/KrishnaDN/x-vector-pytorch#installation
- And also on "Spoken Language Recognition using X-vectors" by David Snyder and his partners: https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
